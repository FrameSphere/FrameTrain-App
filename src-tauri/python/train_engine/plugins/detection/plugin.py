"""
plugins/detection/plugin.py
===========================
Objekt-Erkennungs-Plugin für FrameTrain.

Unterstützt:
  - YOLO v5/v8/v10/v11 (via ultralytics — bevorzugt)
  - Faster R-CNN       (via torchvision — Fallback)

Dataset-Formate:
  - YOLO: data.yaml + images/train/ + labels/train/
  - COCO: annotations.json (Faster-RCNN)

metrics.json Format: flat auf Root-Level (Rust-kompatibel).
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import TrainingConfig
from core.metrics import MetricsCollector
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol


class Plugin(TrainPlugin):
    """Objekt-Erkennungs-Plugin — YOLO (bevorzugt) oder Faster-RCNN."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._start_time = time.time()
        self.backend: str = "unknown"
        self.data_yaml: Optional[str] = None
        self.num_classes: int = 0
        self.class_names: List[str] = []
        self._yolo_model = None
        self._yolo_results = None
        self._rcnn_trainer = None
        self._rcnn_loader = None
        self.metrics = MetricsCollector()
        self._global_step = 0

    # ── setup ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        import random, torch
        import numpy as np

        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        if torch.cuda.is_available():
            MessageProtocol.status("device", f"GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            MessageProtocol.status("device", "Apple Silicon GPU (MPS)")
        else:
            MessageProtocol.status("device", "CPU")

        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)

        try:
            import ultralytics  # noqa: F401
            self.backend = "yolo"
        except ImportError:
            self.backend = "faster_rcnn"
            MessageProtocol.warning(
                "ultralytics nicht installiert — nutze Faster-RCNN.\n"
                "Für YOLO: pip install ultralytics"
            )

    # ── load_data ─────────────────────────────────────────────────────────

    def load_data(self) -> None:
        cfg  = self.config
        root = Path(cfg.dataset_path)

        if self.backend == "yolo":
            candidates = list(root.glob("*.yaml")) + list(root.glob("*.yml"))
            if not candidates:
                raise FileNotFoundError(
                    f"Kein data.yaml in: {root}\n"
                    "YOLO-Datensatz braucht eine data.yaml Datei."
                )
            self.data_yaml = str(candidates[0])
            MessageProtocol.status("loading", f"YOLO data.yaml: {self.data_yaml}")

            try:
                import yaml
                with open(self.data_yaml, "r") as f:
                    y = yaml.safe_load(f)
                self.class_names = y.get("names", [])
                self.num_classes = y.get("nc", len(self.class_names))
                MessageProtocol.status(
                    "loading",
                    f"{self.num_classes} Klassen: "
                    f"{', '.join(self.class_names[:5])}{'...' if len(self.class_names) > 5 else ''}"
                )
            except Exception as e:
                MessageProtocol.warning(f"data.yaml konnte nicht geparst werden: {e}")
        else:
            classes_file = root / "classes.txt"
            if classes_file.exists():
                self.class_names = [l.strip() for l in classes_file.read_text().splitlines() if l.strip()]
                self.num_classes = len(self.class_names) + 1
            else:
                self.num_classes = 91
            MessageProtocol.status("loading", f"{self.num_classes} Klassen (inkl. Hintergrund)")
            self._build_rcnn_loader(root)

    def _build_rcnn_loader(self, root: Path) -> None:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        from PIL import Image

        cfg          = self.config
        train_imgs   = root / "images" / "train"
        train_labels = root / "labels"  / "train"
        if not train_imgs.exists():
            train_imgs = root / "train"

        img_files = sorted([
            f for f in train_imgs.glob("*")
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        class _DS(Dataset):
            def __init__(self, imgs, lbls, tf):
                self.imgs = imgs; self.lbls = lbls; self.tf = tf
            def __len__(self): return len(self.imgs)
            def __getitem__(self, idx):
                img_p = self.imgs[idx]
                img   = Image.open(img_p).convert("RGB")
                w, h  = img.size
                lp    = self.lbls / f"{img_p.stem}.txt"
                boxes, labs = [], []
                if lp.exists():
                    for line in lp.read_text().splitlines():
                        parts = line.split()
                        if len(parts) < 5: continue
                        cls = int(parts[0])
                        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        boxes.append([(xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h])
                        labs.append(cls + 1)
                if self.tf: img = self.tf(img)
                bt = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4))
                lt = torch.as_tensor(labs,  dtype=torch.int64)   if labs  else torch.zeros((0,), dtype=torch.int64)
                return img, {"boxes": bt, "labels": lt, "image_id": torch.tensor([idx])}

        tf = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
        ])
        self._rcnn_loader = DataLoader(
            _DS(img_files, train_labels, tf),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=lambda b: list(zip(*b)),
        )
        MessageProtocol.status("loading", f"{len(img_files):,} Trainingsbilder")

    # ── build_model ───────────────────────────────────────────────────────

    def build_model(self) -> None:
        if self.backend == "yolo":
            from ultralytics import YOLO
            self._yolo_model = YOLO(self.config.model_path)
            MessageProtocol.status("loading", f"YOLO geladen: {self.config.model_path}")
        else:
            self._build_rcnn_model()

    def _build_rcnn_model(self) -> None:
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        dev = (torch.device("cuda") if torch.cuda.is_available() else
               torch.device("mps")  if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else
               torch.device("cpu"))

        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_f  = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, self.num_classes)
        model = model.to(dev)

        params = [p for p in model.parameters() if p.requires_grad]
        opt    = torch.optim.SGD(params, lr=self.config.learning_rate,
                                 momentum=self.config.sgd_momentum,
                                 weight_decay=self.config.weight_decay)

        self._rcnn_trainer = {"model": model, "optimizer": opt, "device": dev}
        MessageProtocol.status("loading", "Faster-RCNN geladen (torchvision)")

    # ── train ─────────────────────────────────────────────────────────────

    def train(self) -> None:
        cfg = self.config

        if self.backend == "yolo":
            MessageProtocol.status("training", "YOLO Training gestartet...")
            MessageProtocol.progress(
                epoch=1, total_epochs=cfg.epochs, step=0, total_steps=cfg.epochs,
                train_loss=0.0,
            )
            self._yolo_results = self._yolo_model.train(
                data=self.data_yaml,
                epochs=cfg.epochs,
                batch=cfg.batch_size,
                imgsz=cfg.image_size,
                lr0=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                warmup_epochs=max(1, int(cfg.warmup_ratio * cfg.epochs)),
                seed=cfg.seed,
                project=cfg.effective_output_dir(),
                name="train",
                exist_ok=True,
                verbose=False,
            )
            self._global_step = cfg.epochs
            MessageProtocol.progress(
                epoch=cfg.epochs, total_epochs=cfg.epochs,
                step=cfg.epochs, total_steps=cfg.epochs, train_loss=0.0,
            )

        else:
            MessageProtocol.status("training", "Faster-RCNN Training gestartet...")
            model    = self._rcnn_trainer["model"]
            optimizer = self._rcnn_trainer["optimizer"]
            dev      = self._rcnn_trainer["device"]
            steps    = len(self._rcnn_loader)
            total_steps = cfg.epochs * steps

            for epoch in range(1, cfg.epochs + 1):
                if self.is_stopped: break
                MessageProtocol.status("epoch", f"Epoche {epoch}/{cfg.epochs}")
                model.train()
                epoch_loss = 0.0
                for step, (images, targets) in enumerate(self._rcnn_loader, 1):
                    if self.is_stopped: break
                    images  = [img.to(dev) for img in images]
                    targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    losses    = sum(loss_dict.values())
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    self._global_step += 1
                    epoch_loss += losses.item()
                    if self._global_step % cfg.logging_steps == 0:
                        avg = epoch_loss / step
                        MessageProtocol.progress(
                            epoch=epoch, total_epochs=cfg.epochs,
                            step=self._global_step, total_steps=total_steps,
                            train_loss=avg,
                        )
                        self.metrics.record(epoch, self._global_step, {"train_loss": avg})

        self.metrics.total_epochs = cfg.epochs
        self.metrics.total_steps  = self._global_step

    # ── validate ──────────────────────────────────────────────────────────

    def validate(self) -> Dict[str, float]:
        duration = int(time.time() - self._start_time)
        extra: Dict[str, Any] = {
            "total_epochs":              self.config.epochs,
            "total_steps":               self._global_step,
            "training_duration_seconds": duration,
        }

        if self.backend == "yolo" and self._yolo_model:
            try:
                m = self._yolo_model.val()
                box = getattr(m, "box", m)
                extra["mAP50"]    = float(getattr(box, "map50", 0.0))
                extra["mAP50-95"] = float(getattr(box, "map",   0.0))
            except Exception as e:
                MessageProtocol.warning(f"YOLO Validierung fehlgeschlagen: {e}")

        return self.metrics.final_metrics(extra)

    # ── export ────────────────────────────────────────────────────────────

    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)

        if self.backend == "yolo":
            run_dir = Path(self.config.effective_output_dir()) / "train"
            best    = run_dir / "weights" / "best.pt"
            if best.exists():
                import shutil
                shutil.copy(best, out / "best.pt")
                MessageProtocol.status("saving", "best.pt kopiert")
            else:
                MessageProtocol.warning("best.pt nicht gefunden")
        else:
            import torch
            torch.save(self._rcnn_trainer["model"].state_dict(), out / "model.pth")

        meta = {
            "backend":     self.backend,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }
        (out / "config.json").write_text(json.dumps(meta, indent=2))

        # metrics.json flat (Rust-kompatibel)
        final = self.validate()
        self.metrics.save_with_overrides(str(out), final)

        MessageProtocol.status("saved", f"Modell gespeichert: {out}")
        return str(out)
