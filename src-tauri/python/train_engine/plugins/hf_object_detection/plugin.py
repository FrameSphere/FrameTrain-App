"""Objekterkennung mit HuggingFace-Detektoren — task_type 'hf_object_detection'

Trainiert, was AutoModelForObjectDetection laedt: DETR, Conditional und
Deformable DETR, RT-DETR/RT-DETRv2 im HF-Format (PekingU/rtdetr_*), YOLOS,
DAB-DETR, D-FINE. Ultralytics-Gewichte (yolo11n.pt, rtdetr-l.pt) gehoeren
zum YOLO-Plugin.

Daten: YOLO-Boxen (images/ + labels/*.txt) oder COCO-JSON. Beides wird intern
in COCO-Annotationen umgerechnet (ft_data.od_formats), die der Bildprozessor
des Modells in sein eigenes Format bringt. Der Klassifikationskopf wird auf
die Klassen des Datasets neu angelegt (ignore_mismatched_sizes); Backbone und
Decoder behalten ihre Gewichte.

Eigene Trainingsschleife statt HF-Trainer: Detektoren brauchen Listen von
Label-Dicts im Batch, und die mAP je Epoche entsteht aus Vorhersagen, nicht
aus Logits — beides passt schlecht in Trainer.compute_metrics.
"""
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from ft_data import hf_detection as hfd
from ft_data.od_formats import hflip_record, load_detection_dataset, to_coco_target
from ft_data.od_metrics import detection_metrics


def split_records(splits: Dict[str, list], seed: int = 42, val_fraction: float = 0.2):
    """(train, val): vorhandenes val/ nutzen, sonst 20 % von train abzweigen."""
    train = list(splits.get("train") or [])
    val = list(splits.get("val") or [])
    if not train and splits.get("test"):
        train = list(splits["test"])
    if not val and len(train) >= 2:
        rng = random.Random(seed)
        idx = list(range(len(train)))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(train) * val_fraction)))
        val_idx = set(idx[:n_val])
        val = [r for i, r in enumerate(train) if i in val_idx]
        train = [r for i, r in enumerate(train) if i not in val_idx]
    return train, val


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        pc = config.plugin_config or {}
        self.imgsz = int(pc.get("imgsz", 512) or 0)
        self.augment = bool(pc.get("augment", True))
        self.backbone_lr = float(pc.get("backbone_lr", 0) or 0)
        self.conf = float(pc.get("conf", 0.25))
        self.device_arg = str(pc.get("device", "") or "")
        self.model = None
        self.processor = None
        self.model_type = ""
        self.classes: List[str] = []
        self.train_records: List[dict] = []
        self.val_records: List[dict] = []
        self.data_format = ""
        self.device = "cpu"
        self._metrics: Dict[str, float] = {}
        self._best_map = -1.0
        self._best_state: Optional[dict] = None
        self._last_train_loss: Optional[float] = None
        self._last_val_loss: Optional[float] = None
        self._best_epoch = 0
        self._steps_done = 0

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> bool:
        problem = hfd.check_model_dir(Path(self.config.model_path))
        if problem:
            MessageProtocol.error("Kein HuggingFace-Detektor", problem)
            return False
        self.model_type = hfd.read_model_type(Path(self.config.model_path))
        from transformers import AutoImageProcessor
        MessageProtocol.status("init", f"Architektur: {hfd.SUPPORTED_MODEL_TYPES[self.model_type]} | Lade Bildprozessor...")
        self.processor = AutoImageProcessor.from_pretrained(self.config.model_path)
        hfd.configure_processor_size(self.processor, self.imgsz)
        return True

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def load_data(self) -> bool:
        root = Path(self.config.dataset_path or "")
        if not root.exists():
            MessageProtocol.error("Dataset nicht gefunden", f"Pfad: {root}")
            return False
        try:
            ds = load_detection_dataset(root)
        except ValueError as e:
            MessageProtocol.error("Dataset nicht lesbar", str(e))
            return False
        self.data_format = ds["format"]
        self.classes = ds["classes"]
        self.train_records, self.val_records = split_records(ds["splits"], int(self.config.seed or 42))
        n_boxes = sum(len(r["boxes"]) for r in self.train_records)
        if not self.train_records or n_boxes == 0:
            MessageProtocol.error(
                "Keine Boxen im Trainings-Split",
                f"In {root} wurden {len(self.train_records)} Trainingsbilder, aber keine einzige Box gefunden. "
                "Erwartet: images/ + labels/*.txt (klasse cx cy w h) oder eine COCO-JSON-Datei.")
            return False
        if self.config.max_eval_samples and len(self.val_records) > self.config.max_eval_samples:
            self.val_records = self.val_records[: int(self.config.max_eval_samples)]
        MessageProtocol.status("data",
            f"{self.data_format}: {len(self.classes)} Klassen, {len(self.train_records)} Trainings- und "
            f"{len(self.val_records)} Validierungsbilder, {n_boxes} Boxen im Training")
        return True

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> bool:
        from transformers import AutoModelForObjectDetection
        id2label = {i: n for i, n in enumerate(self.classes)}
        label2id = {n: i for i, n in id2label.items()}
        MessageProtocol.status("model", f"Lade {Path(self.config.model_path).name} mit {len(self.classes)} Klassen...")
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.config.model_path, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
        self.device = hfd.pick_device(self.device_arg)
        self.model.to(self.device)
        return True

    # ── 4. Training ─────────────────────────────────────────────────────────
    def _batch(self, records: List[dict], train: bool) -> Dict[str, Any]:
        images, targets = [], []
        for i, rec in enumerate(records):
            img = hfd.load_image(rec["path"])
            if train and self.augment and random.random() < 0.5:
                from PIL import ImageOps
                img = ImageOps.mirror(img)
                rec = hflip_record(rec)
            images.append(img)
            targets.append(to_coco_target(rec, i))
        enc = self.processor(images=images, annotations=targets, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in enc.items() if k != "labels" and hasattr(v, "to")}
        batch["labels"] = [{k: v.to(self.device) for k, v in lab.items()} for lab in enc["labels"]]
        return batch

    def _optimizer(self):
        import torch
        lr = float(self.config.learning_rate or 1e-4)
        # Backbone langsamer als Kopf/Decoder (DETR-Rezept: 1e-5 zu 1e-4).
        blr = self.backbone_lr or lr / 10
        head, backbone = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (backbone if "backbone" in name else head).append(p)
        groups = [{"params": head, "lr": lr}]
        if backbone:
            groups.append({"params": backbone, "lr": blr})
        return torch.optim.AdamW(groups, lr=lr, weight_decay=float(self.config.weight_decay or 1e-4))

    def train(self) -> bool:
        import torch
        bs = max(1, int(self.config.batch_size or 4))
        epochs = max(1, int(self.config.epochs or 1))
        steps_per_epoch = math.ceil(len(self.train_records) / bs)
        total_steps = steps_per_epoch * epochs
        if self.config.max_steps and self.config.max_steps > 0:
            total_steps = min(total_steps, int(self.config.max_steps))
        opt = self._optimizer()
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1 - s / max(1, total_steps)))
        clip = float(self.config.max_grad_norm or 0.1)
        rng = random.Random(int(self.config.seed or 42))
        random.seed(int(self.config.seed or 42))

        for epoch in range(1, epochs + 1):
            self.model.train()
            order = list(self.train_records)
            rng.shuffle(order)
            losses: List[float] = []
            for b in range(steps_per_epoch):
                if self.is_stopped or self._steps_done >= total_steps:
                    break
                chunk = order[b * bs:(b + 1) * bs]
                if not chunk:
                    break
                try:
                    out = self.model(**self._batch(chunk, train=True))
                except NotImplementedError as e:
                    if self.device != "mps":
                        raise
                    # Manche Detektor-Operationen fehlen auf MPS (z.B. deformierbare Attention).
                    MessageProtocol.warning(f"Auf MPS nicht verfuegbar ({e}); weiter auf CPU.")
                    self.device = "cpu"
                    self.model.to("cpu")
                    out = self.model(**self._batch(chunk, train=True))
                loss = out.loss
                if not torch.isfinite(loss):
                    MessageProtocol.warning(f"Loss nicht endlich in Schritt {self._steps_done + 1}; Schritt uebersprungen.")
                    opt.zero_grad()
                    continue
                opt.zero_grad()
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                opt.step()
                sched.step()
                self._steps_done += 1
                losses.append(float(loss.item()))
                MessageProtocol.progress(
                    epoch=epoch, total_epochs=epochs, step=self._steps_done, total_steps=total_steps,
                    train_loss=sum(losses) / len(losses), learning_rate=float(opt.param_groups[0]["lr"]))
            if self.is_stopped:
                return True
            self._last_train_loss = sum(losses) / len(losses) if losses else None
            metrics, val_loss = self._evaluate()
            self._last_val_loss = val_loss
            self._metrics = metrics
            if metrics.get("mAP50-95", 0.0) > self._best_map or self._best_state is None:
                self._best_map = metrics.get("mAP50-95", 0.0)
                self._best_epoch = epoch
                self._best_state = {k: v.detach().to("cpu").clone() for k, v in self.model.state_dict().items()}
            MessageProtocol.progress(
                epoch=epoch, total_epochs=epochs, step=self._steps_done, total_steps=total_steps,
                train_loss=self._last_train_loss or 0.0, val_loss=val_loss,
                learning_rate=float(opt.param_groups[0]["lr"]), metrics=metrics)
            MessageProtocol.status("train", f"[Metric] epoch={epoch}/{epochs} loss={self._last_train_loss or 0:.4f} "
                                   + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if self._steps_done >= total_steps:
                break
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return True

    def _evaluate(self):
        """(Kennzahlen, Val-Loss) auf dem Validierungs-Split."""
        import torch
        if not self.val_records:
            return {}, None
        self.model.eval()
        bs = max(1, int(self.config.batch_size or 4))
        preds, gts, losses = [], [], []
        for i in range(0, len(self.val_records), bs):
            chunk = self.val_records[i:i + bs]
            with torch.no_grad():
                out = self.model(**self._batch(chunk, train=False))
            if out.loss is not None and torch.isfinite(out.loss):
                losses.append(float(out.loss.item()))
            images = [hfd.load_image(r["path"]) for r in chunk]
            for dets, rec in zip(hfd.predict(self.model, self.processor, images, self.device), chunk):
                preds.append(hfd.top_k(dets))
                gts.append(rec["boxes"])
        metrics = detection_metrics(preds, gts, conf=self.conf)
        return metrics, (sum(losses) / len(losses) if losses else None)

    # ── 5. Validierung ──────────────────────────────────────────────────────
    def validate(self) -> Dict[str, float]:
        return dict(self._metrics)

    def get_metrics(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "framework": "transformers", "task": "hf_object_detection",
            "architecture": self.model_type, "device": self.device,
            "num_labels": len(self.classes), "dataset_format": self.data_format,
            "n_train": len(self.train_records), "n_val": len(self.val_records),
            "total_steps": self._steps_done, "best_epoch": self._best_epoch,
        }
        if self._last_train_loss is not None:
            out["final_train_loss"] = self._last_train_loss
        if self._last_val_loss is not None:
            out["final_val_loss"] = self._last_val_loss
        return out

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.processor.save_pretrained(str(out))
        (out / "label_mapping.json").write_text(
            json.dumps({"classes": self.classes}, indent=2, ensure_ascii=False), encoding="utf-8")
        # Die App liest "task" aus model.json, um beim Testen das Plugin zu waehlen.
        (out / "model.json").write_text(json.dumps({
            "framework": "transformers", "task": "hf_object_detection", "model_type": self.model_type,
            "classes": self.classes, "imgsz": self.imgsz, "metrics": self._metrics,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)
