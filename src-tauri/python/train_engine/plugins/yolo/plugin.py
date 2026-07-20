"""YOLO Object Detection Plugin — task_type: 'detect'"""
import json, os, shutil
from pathlib import Path
from typing import Any, Dict, Optional
from core.config import TrainingConfig
from core.protocol import MessageProtocol


class YOLOPlugin:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.is_stopped = False
        self.results = None
        self._yaml_path: Optional[str] = None
        self._output_dir: Optional[Path] = None
        pc = config.plugin_config or {}
        self.yolo_model   = pc.get("yolo_model",   "yolov8n.pt")
        self.task         = pc.get("task",          "detect")
        self.imgsz        = int(pc.get("imgsz",    640))
        self.patience     = int(pc.get("patience",  50))
        self.augment      = bool(pc.get("augment",  True))
        self.optimizer_name = pc.get("optimizer",  "SGD")
        self.lr0          = float(pc.get("lr0",     0.01))
        self.lrf          = float(pc.get("lrf",     0.01))
        self.momentum     = float(pc.get("momentum", 0.937))
        self.wd           = float(pc.get("weight_decay", 0.0005))
        self.device_arg   = pc.get("device", "")

    def setup(self) -> bool:
        try:
            from ultralytics import YOLO  # noqa
        except ImportError:
            MessageProtocol.error("Ultralytics nicht installiert",
                "pip install ultralytics>=8.0.0")
            return False
        dsp = self.config.dataset_path
        if not dsp or not Path(dsp).exists():
            MessageProtocol.error("Dataset nicht gefunden", f"Pfad: {dsp!r}")
            return False
        yaml_path = self._find_or_build_yaml(Path(dsp))
        if yaml_path is None:
            return False
        self._yaml_path = str(yaml_path)
        self._output_dir = Path(self.config.output_path)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        MessageProtocol.status("setup",
            f"YOLO Setup OK\n  Model: {self.yolo_model}\n  Task: {self.task}\n  YAML: {self._yaml_path}")
        return True

    def _find_or_build_yaml(self, root: Path) -> Optional[Path]:
        for c in [root/"dataset.yaml", root/"data.yaml"]:
            if c.exists():
                return c
        train_dir = root / "train"
        if train_dir.is_dir():
            for c in [train_dir.parent/"dataset.yaml", train_dir.parent/"data.yaml"]:
                if c.exists():
                    return c
        MessageProtocol.status("setup", "Kein dataset.yaml — generiere automatisch...")
        return self._generate_yaml(root)

    def _generate_yaml(self, root: Path) -> Optional[Path]:
        def find_images(base: Path) -> Optional[Path]:
            for c in [base/"images", base]:
                if c.is_dir() and any(f.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp")
                    for f in c.rglob("*") if f.is_file()):
                    return c
            return None
        train_imgs = find_images(root/"train") or find_images(root/"images"/"train")
        val_imgs = (find_images(root/"val") or find_images(root/"images"/"val")
                    or find_images(root/"valid") or find_images(root/"images"/"valid"))
        if not train_imgs:
            MessageProtocol.error("YAML-Generierung fehlgeschlagen",
                f"Keine Trainings-Bilder in {root}.\nErwartet: train/images/*.jpg + train/labels/*.txt")
            return None
        classes = self._detect_classes(root) or ["object"]
        yaml_path = root / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {root.resolve()}\n")
            f.write(f"train: {train_imgs.resolve()}\n")
            f.write(f"val: {val_imgs.resolve() if val_imgs else train_imgs.resolve()}\n")
            f.write(f"nc: {len(classes)}\n")
            f.write(f"names: {classes}\n")
        MessageProtocol.status("setup", f"dataset.yaml generiert: {len(classes)} Klassen")
        return yaml_path

    def _detect_classes(self, root: Path) -> list:
        class_ids = set()
        for ld in [root/"labels", root/"train"/"labels", root/"val"/"labels"]:
            if not ld.is_dir(): continue
            for txt in ld.rglob("*.txt"):
                try:
                    for line in txt.read_text(encoding="utf-8").strip().splitlines():
                        parts = line.strip().split()
                        if parts: class_ids.add(int(parts[0]))
                except Exception: continue
        for nf in [root/"classes.txt", root/"obj.names", root/"labels.txt"]:
            if nf.exists():
                names = [l.strip() for l in nf.read_text(encoding="utf-8").splitlines() if l.strip()]
                if names: return names
        return [f"class_{i}" for i in sorted(class_ids)] if class_ids else []

    def load_data(self) -> None: pass
    def build_model(self) -> None: pass

    def train(self) -> bool:
        if not self._yaml_path:
            MessageProtocol.error("Training", "setup() nicht aufgerufen.")
            return False
        try:
            from ultralytics import YOLO
        except ImportError:
            MessageProtocol.error("Ultralytics", "Import fehlgeschlagen.")
            return False
        try:
            import torch
            MessageProtocol.status("train", f"Lade {self.yolo_model}...")
            self.model = YOLO(self.yolo_model)
            if self.device_arg:
                device = self.device_arg
            elif torch.cuda.is_available():
                device = "0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            def on_epoch_end(trainer):
                if self.is_stopped:
                    trainer.stop = True
                    return
                m = trainer.metrics or {}
                ep = trainer.epoch + 1
                tot = trainer.epochs
                loss = float(m.get("train/box_loss", 0) + m.get("train/cls_loss", 0))
                map50 = float(m.get("metrics/mAP50(B)", 0))
                map5095 = float(m.get("metrics/mAP50-95(B)", 0))
                MessageProtocol.progress(epoch=ep, total_epochs=tot, step=ep, total_steps=tot,
                    train_loss=loss, metrics={"mAP50": map50, "mAP50-95": map5095})
                MessageProtocol.status("train",
                    f"[Metric] epoch={ep}/{tot} loss={loss:.4f} mAP50={map50:.4f}")

            self.model.add_callback("on_train_epoch_end", on_epoch_end)
            self.results = self.model.train(
                data=self._yaml_path,
                epochs=self.config.epochs,
                batch=self.config.batch_size,
                imgsz=self.imgsz,
                device=device,
                patience=self.patience,
                augment=self.augment,
                optimizer=self.optimizer_name,
                lr0=self.lr0, lrf=self.lrf,
                momentum=self.momentum,
                weight_decay=self.wd,
                project=str(self._output_dir),
                name="train",
                exist_ok=True,
                verbose=False, plots=False, save=True,
            )
            return True
        except Exception as e:
            import traceback
            MessageProtocol.error("YOLO Training Fehler", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            return False

    def validate(self) -> Dict[str, float]:
        if not self.results: return {}
        try:
            m = self.results.results_dict or {}
            return {
                "mAP50":    float(m.get("metrics/mAP50(B)",    0)),
                "mAP50-95": float(m.get("metrics/mAP50-95(B)", 0)),
                "precision": float(m.get("metrics/precision(B)", 0)),
                "recall":    float(m.get("metrics/recall(B)",    0)),
            }
        except Exception: return {}

    def save_model(self, output_path: str, **_) -> bool:
        try:
            run_dir = self._output_dir / "train"
            best = run_dir / "weights" / "best.pt"
            last = run_dir / "weights" / "last.pt"
            src = best if best.exists() else last if last.exists() else None
            if not src:
                MessageProtocol.error("Save", f"Kein Checkpoint in {run_dir/'weights'}"); return False
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), output_path)
            meta = {"framework":"ultralytics","base_model":self.yolo_model,
                    "task":self.task,"imgsz":self.imgsz,"yaml_path":self._yaml_path,
                    "metrics":self.validate()}
            with open(Path(output_path).with_suffix(".json"), "w") as f:
                json.dump(meta, f, indent=2)
            MessageProtocol.status("save", f"YOLO-Modell gespeichert: {output_path}")
            return True
        except Exception as e:
            MessageProtocol.error("Save Fehler", str(e)); return False

    def get_metrics(self) -> Dict[str, Any]:
        return {"framework":"ultralytics","base_model":self.yolo_model,"task":self.task,**self.validate()}

    def export(self) -> str:
        try:
            out = Path(self.config.output_path) / "model.pt"
            self.save_model(str(out))
            return str(self.config.output_path)
        except Exception as e:
            MessageProtocol.error("Export Fehler", str(e))
            return str(self.config.output_path)
