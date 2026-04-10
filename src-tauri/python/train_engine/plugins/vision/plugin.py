"""
plugins/vision/plugin.py
========================
Vision-Plugin für FrameTrain — Bildklassifikation.

Unterstützt:
  - ImageFolder-Struktur (train/klasse1/*.jpg, train/klasse2/*.jpg)
  - Annotations-JSON  ({"image": "x.jpg", "label": "katze"})
  - Modelle via timm (ViT, ResNet, EfficientNet, ConvNeXt, Swin, ...)
  - Modelle via torchvision (als Fallback)

Aufgerufen vom Orchestrator via TrainPlugin-Interface.
metrics.json Format: flat auf Root-Level (Rust-kompatibel).
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import TrainingConfig
from core.dataset_manager import IMAGE_EXTENSIONS, detect_image_dataset_structure
from core.metrics import MetricsCollector
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol


class _FlatAnnotatedDataset:
    """Lädt Bilder aus einem Flat-Ordner mit annotations.json."""

    def __init__(self, root: Path, transform=None):
        from PIL import Image
        ann_file = root / "annotations.json"
        with open(ann_file, "r") as f:
            annotations = json.load(f)

        all_labels = sorted({item["label"] for item in annotations})
        self.label_map = {name: i for i, name in enumerate(all_labels)}
        self.class_names = all_labels

        self.samples: List[Tuple[Path, int]] = [
            (root / item["image"], self.label_map[item["label"]])
            for item in annotations
            if (root / item["image"]).exists()
        ]
        self.transform = transform
        self._Image = Image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self._Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class Plugin(TrainPlugin):
    """Bildklassifikations-Plugin mit timm/torchvision."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._start_time = time.time()
        self.device = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.num_classes = 0
        self.class_names: List[str] = []
        self.metrics = MetricsCollector()
        self._global_step = 0

    def setup(self) -> None:
        import torch, random
        import numpy as np

        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            MessageProtocol.status("device", f"GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            MessageProtocol.status("device", "Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            MessageProtocol.status("device", "CPU")

        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms

        cfg  = self.config
        root = Path(cfg.dataset_path)
        size = cfg.image_size

        train_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        structure  = detect_image_dataset_structure(str(root))
        train_path = root / "train" if (root / "train").exists() else root
        val_path   = root / "val"   if (root / "val").exists()   else None

        MessageProtocol.status("loading", f"Datensatz-Struktur: {structure}")

        if structure == "imagefolder":
            from torchvision.datasets import ImageFolder
            train_ds = ImageFolder(str(train_path), transform=train_tf)
            self.num_classes = len(train_ds.classes)
            self.class_names = train_ds.classes
            val_ds = ImageFolder(str(val_path), transform=val_tf) if val_path and val_path.exists() else None
        else:
            train_ds = _FlatAnnotatedDataset(train_path, transform=train_tf)
            self.num_classes = len(train_ds.class_names)
            self.class_names = train_ds.class_names
            val_ds = _FlatAnnotatedDataset(val_path, transform=val_tf) \
                if val_path and (val_path / "annotations.json").exists() else None

        MessageProtocol.status(
            "loading",
            f"Datensatz: {len(train_ds):,} Bilder | {self.num_classes} Klassen"
        )

        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            drop_last=cfg.dataloader_drop_last,
        )
        if val_ds:
            self.val_loader = DataLoader(
                val_ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            )
            MessageProtocol.status("loading", f"Validierung: {len(val_ds):,} Bilder")

    def build_model(self) -> None:
        import torch
        import torch.nn as nn

        cfg        = self.config
        model_name = cfg.model_path
        n_cls      = self.num_classes if self.num_classes > 0 else 1000

        try:
            import timm
            if Path(model_name).exists():
                meta_file = Path(model_name).parent / "config.json"
                arch = "resnet50"
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text())
                    arch = meta.get("model_architecture", "resnet50")
                self.model = timm.create_model(arch, pretrained=False, num_classes=n_cls)
                state = torch.load(model_name, map_location="cpu")
                self.model.load_state_dict(state, strict=False)
                MessageProtocol.status("loading", f"Checkpoint geladen: {model_name}")
            else:
                self.model = timm.create_model(model_name, pretrained=True, num_classes=n_cls)
                MessageProtocol.status("loading", f"timm-Modell: {model_name}")
        except (ImportError, Exception) as e:
            MessageProtocol.warning(f"timm nicht verfügbar ({e}), nutze torchvision ResNet50")
            import torchvision.models as tv
            self.model = tv.resnet50(weights="IMAGENET1K_V2")
            self.model.fc = nn.Linear(self.model.fc.in_features, n_cls)

        n_params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status("loading", f"Modell: {n_params / 1e6:.1f}M Parameter")
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        if cfg.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=cfg.learning_rate,
                momentum=cfg.sgd_momentum, weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_epsilon,
            )

        self._build_scheduler()

    def _build_scheduler(self):
        import torch
        cfg         = self.config
        total_steps = cfg.epochs * (len(self.train_loader) if self.train_loader else 1)
        sched       = cfg.scheduler.lower()

        if sched == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=cfg.cosine_min_lr)
        elif sched in ("step", "steplr"):
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)
        elif sched in ("exponential", "exponentiallr"):
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=cfg.scheduler_gamma)
        else:
            warmup = cfg.warmup_steps or int(total_steps * cfg.warmup_ratio)
            def lr_lambda(step):
                if step < warmup:
                    return step / max(warmup, 1)
                progress = (step - warmup) / max(total_steps - warmup, 1)
                return max(0.0, 1.0 - progress)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self) -> None:
        import torch

        cfg         = self.config
        total_steps = cfg.epochs * len(self.train_loader)

        for epoch in range(1, cfg.epochs + 1):
            if self.is_stopped:
                break

            self.model.train()
            epoch_loss = 0.0
            correct    = 0
            total      = 0

            MessageProtocol.status("epoch", f"Epoche {epoch}/{cfg.epochs}")

            for step, (images, labels) in enumerate(self.train_loader, 1):
                if self.is_stopped:
                    break

                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)
                loss.backward()

                if cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                self._global_step += 1
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total   += labels.size(0)

                if self._global_step % cfg.logging_steps == 0:
                    avg_loss = epoch_loss / step
                    acc      = correct / total
                    lr       = self.optimizer.param_groups[0]["lr"]
                    MessageProtocol.progress(
                        epoch=epoch, total_epochs=cfg.epochs,
                        step=self._global_step, total_steps=total_steps,
                        train_loss=avg_loss, learning_rate=lr,
                        metrics={"accuracy": round(acc, 4)},
                    )
                    self.metrics.record(epoch, self._global_step, {
                        "train_loss": avg_loss, "accuracy": acc, "learning_rate": lr,
                    })

            # Epoch-Checkpoint
            if cfg.save_strategy == "epoch":
                ckpt = Path(cfg.effective_output_dir()) / f"checkpoint-epoch{epoch}"
                ckpt.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), ckpt / "model.pth")
                MessageProtocol.checkpoint(
                    step=self._global_step, path=str(ckpt), epoch=epoch)

            # Validierung
            val_loss, val_acc = self._run_validation()
            if val_loss is not None:
                avg_epoch_loss = epoch_loss / len(self.train_loader)
                MessageProtocol.progress(
                    epoch=epoch, total_epochs=cfg.epochs,
                    step=self._global_step, total_steps=total_steps,
                    train_loss=avg_epoch_loss, val_loss=val_loss,
                    metrics={"val_accuracy": round(val_acc, 4)},
                )
                self.metrics.record(epoch, self._global_step, {
                    "val_loss": val_loss, "val_accuracy": val_acc,
                })

        # Tracker aktualisieren
        self.metrics.total_epochs = self.config.epochs
        self.metrics.total_steps  = self._global_step

    def _run_validation(self) -> Tuple[Optional[float], float]:
        if not self.val_loader:
            return None, 0.0
        import torch
        self.model.eval()
        t_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                out    = self.model(images)
                t_loss += self.criterion(out, labels).item()
                _, pred = out.max(1)
                correct += pred.eq(labels).sum().item()
                total   += labels.size(0)
        self.model.train()
        return (t_loss / len(self.val_loader), correct / max(total, 1))

    def validate(self) -> Dict[str, float]:
        val_loss, val_acc = self._run_validation()
        duration = int(time.time() - self._start_time)
        return self.metrics.final_metrics({
            "total_epochs":              self.config.epochs,
            "total_steps":               self._global_step,
            "training_duration_seconds": duration,
            "val_loss":                  val_loss,
            "val_accuracy":              val_acc,
            "num_classes":               self.num_classes,
        })

    def export(self) -> str:
        import torch
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), out / "model.pth")

        meta = {
            "model_architecture": self.config.model_path,
            "num_classes":  self.num_classes,
            "class_names":  self.class_names,
            "image_size":   self.config.image_size,
        }
        (out / "config.json").write_text(json.dumps(meta, indent=2))

        # metrics.json flat (Rust-kompatibel)
        final = self.validate()
        self.metrics.save_with_overrides(str(out), final)

        MessageProtocol.status("saved", f"Modell gespeichert: {out}")
        return str(out)
