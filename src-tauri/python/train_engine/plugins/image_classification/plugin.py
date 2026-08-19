"""Image Classification Plugin — task_type: 'image_classification'"""
import json, shutil, time, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from core.config import TrainingConfig
from core.protocol import MessageProtocol


def _build_backbone(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    try:
        from torchvision import models
        w = "DEFAULT" if pretrained else None
        a = arch.lower()
        if a == "resnet18":
            m = models.resnet18(weights=w); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
        if a == "resnet50":
            m = models.resnet50(weights=w); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
        if a == "efficientnet_b0":
            m = models.efficientnet_b0(weights=w)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes); return m
        if a == "efficientnet_b4":
            m = models.efficientnet_b4(weights=w)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes); return m
        if a == "vit_b_16":
            m = models.vit_b_16(weights=w)
            m.heads.head = nn.Linear(m.heads.head.in_features, num_classes); return m
        if a in ("mobilenet_v3_small", "mobilenet_v3_large"):
            m = (models.mobilenet_v3_small if a == "mobilenet_v3_small" else models.mobilenet_v3_large)(weights=w)
            m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes); return m
        raise ValueError(f"Unbekannte Architektur: '{arch}'. Unterstuetzt: resnet18, resnet50, efficientnet_b0, efficientnet_b4, vit_b_16, mobilenet_v3_small, mobilenet_v3_large")
    except ImportError:
        raise ImportError("torchvision nicht installiert. pip install torchvision")


def _freeze_base(model: nn.Module, arch: str) -> None:
    for p in model.parameters(): p.requires_grad = False
    a = arch.lower()
    head = (model.fc if "resnet" in a else model.classifier if "efficientnet" in a or "mobilenet" in a else model.heads if "vit" in a else None)
    if head:
        for p in head.parameters(): p.requires_grad = True


def _build_transforms(sz: int, augment: bool, training: bool):
    from torchvision import transforms
    if training and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(sz, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _load_datasets(root: Path, sz: int, augment: bool, batch: int) -> Tuple[DataLoader, DataLoader, List[str]]:
    from torchvision import datasets
    train_dir = root / "train"
    val_dir = root / "val" if (root / "val").is_dir() else root / "valid"

    if train_dir.is_dir() and val_dir.is_dir():
        try:
            ds_tr = datasets.ImageFolder(str(train_dir), transform=_build_transforms(sz, augment, True))
            ds_va = datasets.ImageFolder(str(val_dir),   transform=_build_transforms(sz, False, False))
        except FileNotFoundError as e:
            raise ValueError(f"ImageFolder-Fehler: {e}. Erwartet: Unterordner pro Klasse in train/ und val/.")
        return (DataLoader(ds_tr, batch, shuffle=True, num_workers=0, pin_memory=False),
                DataLoader(ds_va, batch, shuffle=False, num_workers=0, pin_memory=False),
                ds_tr.classes)

    if not any(d.is_dir() for d in root.iterdir()):
        raise ValueError(f"Keine Unterordner in {root}. Erwartet: ein Unterordner pro Klasse.")
    try:
        ds_full = datasets.ImageFolder(str(root), transform=_build_transforms(sz, augment, True))
    except Exception as e:
        raise ValueError(f"ImageFolder-Fehler: {e}")
    if len(ds_full) == 0:
        raise ValueError(f"Keine Bilder in {root}.")
    n_tr = int(len(ds_full) * 0.8)
    n_va = len(ds_full) - n_tr
    ds_tr, ds_va = random_split(ds_full, [n_tr, n_va])
    ds_va_nfm = Subset(datasets.ImageFolder(str(root), transform=_build_transforms(sz, False, False)), ds_va.indices)
    return (DataLoader(ds_tr,    batch, shuffle=True,  num_workers=0, pin_memory=False),
            DataLoader(ds_va_nfm, batch, shuffle=False, num_workers=0, pin_memory=False),
            ds_full.classes)


class ImageClassificationPlugin:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.is_stopped = False
        self.model: Optional[nn.Module] = None
        self.classes: List[str] = []
        self._metrics: Dict[str, float] = {}
        self._history: Dict[str, list] = {"epochs": [], "train_losses": [], "val_losses": [], "accuracies": []}
        self.device = ("cuda" if torch.cuda.is_available()
            else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            else "cpu")
        pc = config.plugin_config or {}
        self.arch        = pc.get("arch",         "resnet18")
        self.image_size  = int(pc.get("image_size",  224))
        self.freeze_base = bool(pc.get("freeze_base", True))
        self.unfreeze_at = int(pc.get("unfreeze_at",  -1))
        self.pretrained  = bool(pc.get("pretrained",   True))
        self.augment     = bool(pc.get("augment",      True))

    def setup(self) -> bool:
        dsp = self.config.dataset_path
        if not dsp or not Path(dsp).exists():
            MessageProtocol.error("Dataset nicht gefunden", f"Pfad: {dsp!r}")
            return False
        try:
            _, _, self.classes = _load_datasets(Path(dsp), self.image_size, self.augment, self.config.batch_size)
        except (ValueError, ImportError) as e:
            MessageProtocol.error("Dataset Setup", str(e)); return False
        if len(self.classes) < 2:
            MessageProtocol.error("Zu wenig Klassen", f"{len(self.classes)} Klasse(n) — mindestens 2 benoetigt.")
            return False
        try:
            self.model = _build_backbone(self.arch, len(self.classes), self.pretrained)
        except (ValueError, ImportError) as e:
            MessageProtocol.error("Modell Setup", str(e)); return False
        if self.freeze_base:
            _freeze_base(self.model, self.arch)
            tr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            tot = sum(p.numel() for p in self.model.parameters())
            MessageProtocol.status("setup", f"Base eingefroren — trainierbar: {tr:,}/{tot:,}")
        self.model = self.model.to(self.device)
        MessageProtocol.status("setup",
            f"Image Classification Setup\n  Arch: {self.arch}\n  Klassen: {len(self.classes)}\n"
            f"  ImageSize: {self.image_size}\n  Device: {self.device}")
        return True

    def load_data(self) -> None: pass
    def build_model(self) -> None: pass

    def train(self) -> bool:
        if self.model is None:
            MessageProtocol.error("Training", "setup() nicht aufgerufen."); return False
        try:
            tr_loader, va_loader, self.classes = _load_datasets(
                Path(self.config.dataset_path), self.image_size, self.augment, self.config.batch_size)
            # Fuer die Analyse-Seite: ohne diese Werte stand dort spaeter
            # "0 Steps", "Dauer 0s" und eine unbekannte Architektur.
            self._n_train = len(tr_loader.dataset)
            self._n_val   = len(va_loader.dataset)
            self._start_time = time.time()
            self._total_steps = 0
        except (ValueError, ImportError) as e:
            MessageProtocol.error("Daten laden", str(e)); return False

        epochs    = self.config.epochs
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=self.config.learning_rate * 0.01)
        save_steps = max(self.config.save_steps, 1)

        try:
            # Shape-Test
            sample_x, _ = next(iter(tr_loader))
            with torch.no_grad(): self.model(sample_x[:1].to(self.device))
            MessageProtocol.status("train", f"Shape-Test OK: {list(sample_x.shape[1:])}")
        except Exception as e:
            MessageProtocol.error("Shape-Test", f"{e}\n{traceback.format_exc()}"); return False

        try:
            for epoch in range(epochs):
                if self.is_stopped: break
                if self.freeze_base and self.unfreeze_at > 0 and epoch == self.unfreeze_at:
                    for p in self.model.parameters(): p.requires_grad = True
                    MessageProtocol.status("train", f"Epoche {epoch+1}: Base aufgetaut")

                # Train
                self.model.train()
                total_loss = correct = n = steps = 0
                for bx, by in tr_loader:
                    if self.is_stopped: break
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    out = self.model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    total_loss += loss.item(); correct += (out.argmax(1) == by).sum().item()
                    n += by.size(0); steps += 1

                # Val
                self.model.eval()
                val_loss = val_correct = val_n = top5 = 0
                with torch.no_grad():
                    for vx, vy in va_loader:
                        if self.is_stopped: break
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        vo = self.model(vx)
                        val_loss    += criterion(vo, vy).item()
                        val_correct += (vo.argmax(1) == vy).sum().item()
                        top5        += (vo.topk(min(5, vo.size(1)), dim=1).indices == vy.unsqueeze(1)).any(dim=1).sum().item()
                        val_n       += vy.size(0)

                avg_tr = total_loss / max(steps, 1)
                avg_va = val_loss   / max(len(va_loader), 1)
                acc    = val_correct / max(val_n, 1)
                t5     = top5       / max(val_n, 1)
                lr_now = optimizer.param_groups[0]["lr"]
                scheduler.step()

                self._history["epochs"].append(epoch + 1)
                self._history["train_losses"].append(avg_tr)
                self._history["val_losses"].append(avg_va)
                self._history["accuracies"].append(acc)

                self._total_steps = int(getattr(self, "_total_steps", 0)) + int(steps)
                MessageProtocol.progress(epoch=epoch+1, total_epochs=epochs, step=epoch+1, total_steps=epochs,
                    train_loss=avg_tr, val_loss=avg_va, learning_rate=lr_now,
                    metrics={"accuracy": acc, "top5_accuracy": t5})
                MessageProtocol.status("train",
                    f"[Metric] epoch={epoch+1}/{epochs} loss={avg_tr:.4f} val_loss={avg_va:.4f} acc={acc:.4f} lr={lr_now:.6f}")

                # Checkpoint
                if (epoch + 1) % save_steps == 0 or (epoch + 1) == epochs:
                    ckpt_dir = Path(self.config.checkpoint_dir)
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / f"checkpoint-{epoch+1}"
                    ckpt_path.mkdir(exist_ok=True)
                    torch.save({"epoch": epoch+1, "model_state_dict": self.model.state_dict(),
                        "val_loss": avg_va, "val_accuracy": acc, "classes": self.classes, "arch": self.arch},
                        ckpt_path / "model.pt")
                    # Nur 2 neueste Checkpoints behalten
                    all_ckpts = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                        key=lambda d: int(d.name.split("-")[-1]))
                    for old in all_ckpts[:-2]: shutil.rmtree(old, ignore_errors=True)

            last_train_loss = self._history["train_losses"][-1] if self._history["train_losses"] else 0.0
            last_val_loss   = self._history["val_losses"][-1]   if self._history["val_losses"]   else 0.0
            self._metrics = {
                "accuracy":   self._history["accuracies"][-1]   if self._history["accuracies"]   else 0.0,
                "val_loss":   last_val_loss,
                "train_loss": last_train_loss,
                # Die Analyse-Seite liest diese Namen. Ohne sie zeigte sie
                # "Final Train Loss 0.0000" und "Dauer 0s" fuer jedes
                # Bild-Training — obwohl die Werte vorlagen.
                "final_train_loss": last_train_loss,
                "final_val_loss":   last_val_loss,
                "total_epochs":     len(self._history["train_losses"]),
                "total_steps":      int(getattr(self, "_total_steps", 0)),
                "training_duration_seconds": int(time.time() - getattr(self, "_start_time", time.time())),
                "architecture":     self.arch,
                "num_labels":       len(self.classes),
                "n_train":          int(getattr(self, "_n_train", 0)),
                "n_val":            int(getattr(self, "_n_val", 0)),
                "device":           str(self.device),
            }
            MessageProtocol.status("train", "Image Classification Training abgeschlossen")
            return True
        except Exception as e:
            MessageProtocol.error("Training Fehler", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            return False

    def validate(self) -> Dict[str, float]:
        return self._metrics

    def save_model(self, output_path: str, **_) -> bool:
        try:
            if self.model is None: return False
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": self.model.state_dict(), "arch": self.arch,
                "classes": self.classes, "image_size": self.image_size,
                "metrics": self._metrics, "training_history": self._history}, output_path)
            with open(Path(output_path).with_suffix(".json"), "w") as f:
                json.dump({"framework": "torchvision", "arch": self.arch, "classes": self.classes,
                    "num_classes": len(self.classes), "image_size": self.image_size, "metrics": self._metrics}, f, indent=2)
            MessageProtocol.status("save", f"Modell gespeichert: {output_path}")
            return True
        except Exception as e:
            MessageProtocol.error("Save Fehler", str(e)); return False

    def get_metrics(self) -> Dict[str, Any]:
        return {"framework": "torchvision", "arch": self.arch, "num_classes": len(self.classes),
            "classes": self.classes, **self._metrics, "training_history": self._history}

    def export(self) -> str:
        try:
            out = Path(self.config.output_path)
            out.mkdir(parents=True, exist_ok=True)
            self.save_model(str(out / "model.pt"))
            with open(out / "metrics.json", "w") as f:
                json.dump(self.get_metrics(), f, indent=2, default=str)
            return str(out)
        except Exception as e:
            MessageProtocol.error("Export Fehler", str(e))
            return str(self.config.output_path)
