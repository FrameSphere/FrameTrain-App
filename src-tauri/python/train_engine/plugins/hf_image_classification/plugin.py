"""
Image Classification (HuggingFace)
==================================
Trainiert das Modell, das der Nutzer tatsaechlich heruntergeladen hat.

Abgrenzung zum aelteren `image_classification`-Plugin: jenes baut ein
torchvision-Backbone aus ImageNet-Gewichten und ignoriert den Download
komplett. Wer `google/vit-base-patch16-224` laedt, trainierte dort ein
resnet18. Dieses Plugin laedt die Gewichte aus `model_path`.

Dataset-Layout: ein Ordner pro Klasse, optional in train/ val/ test/.
    <dataset>/train/katze/*.jpg
    <dataset>/val/hund/*.jpg
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


def _class_dirs(root: Path) -> List[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))


def _images_in(d: Path) -> List[Path]:
    return sorted(f for f in d.rglob("*") if f.suffix.lower() in IMAGE_EXTS)


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.processor = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.classes: List[str] = []
        self.model_type = ""
        self.device_used = "cpu"
        self._trainer = None
        self._start_time = time.time()
        self._last_train_loss = 0.0
        self._last_lr = config.learning_rate

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        import json
        from transformers import AutoImageProcessor

        cfg_path = Path(self.config.model_path) / "config.json"
        if cfg_path.exists():
            try:
                self.model_type = json.loads(cfg_path.read_text(encoding="utf-8")).get("model_type", "")
            except Exception:
                self.model_type = ""
        MessageProtocol.status("init", f"✅ Architektur erkannt: {self.model_type or 'unbekannt'} | Lade Bild-Prozessor...")
        self.processor = AutoImageProcessor.from_pretrained(self.config.model_path)
        MessageProtocol.status("init", "Bild-Prozessor geladen ✓")

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def load_data(self) -> None:
        from datasets import Dataset

        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")

        train_root = root / "train" if (root / "train").is_dir() else root
        val_root: Optional[Path] = None
        for name in ("val", "validation", "test"):
            if (root / name).is_dir():
                val_root = root / name
                break

        self.classes = [d.name for d in _class_dirs(train_root)]
        if len(self.classes) < 2:
            raise ValueError(
                f"In '{train_root}' wurden {len(self.classes)} Klassenordner gefunden "
                f"({self.classes or 'keine'}). Fuer eine Bildklassifikation braucht es "
                "mindestens zwei Unterordner — einen pro Klasse."
            )
        label2id = {c: i for i, c in enumerate(self.classes)}

        def collect(base: Path) -> Dict[str, list]:
            paths, labels = [], []
            for d in _class_dirs(base):
                if d.name not in label2id:
                    continue
                for f in _images_in(d):
                    paths.append(str(f))
                    labels.append(label2id[d.name])
            return {"path": paths, "labels": labels}

        train_raw = collect(train_root)
        if not train_raw["path"]:
            raise ValueError(f"Keine Bilddateien unter '{train_root}' gefunden.")
        MessageProtocol.status(
            "loading_data",
            f"Klassen ({len(self.classes)}): {', '.join(self.classes)} | "
            f"{len(train_raw['path'])} Trainingsbilder",
        )

        train_ds = Dataset.from_dict(train_raw)
        if val_root is not None:
            eval_ds = Dataset.from_dict(collect(val_root))
        else:
            split = train_ds.train_test_split(test_size=0.1, seed=self.config.seed)
            train_ds, eval_ds = split["train"], split["test"]
            MessageProtocol.status("loading_data", "Kein val/-Ordner gefunden — 10% des Trainings als Validierung abgetrennt.")

        eval_ds = hft.cap_eval_dataset(eval_ds, getattr(self.config, "max_eval_samples", 0), self.config.seed)

        self.train_dataset = train_ds
        self.eval_dataset = eval_ds
        MessageProtocol.status("loading_data", f"✅ Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        from transformers import AutoModelForImageClassification

        MessageProtocol.status("building_model", "Lade Modell fuer Bildklassifikation...")
        self.model = AutoModelForImageClassification.from_pretrained(
            self.config.model_path,
            num_labels=len(self.classes),
            id2label={i: c for i, c in enumerate(self.classes)},
            label2id={c: i for i, c in enumerate(self.classes)},
            ignore_mismatched_sizes=True,   # Klassifikationskopf wird ersetzt
        )
        params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status(
            "building_model",
            f"✅ Modell geladen | Parameter: {params/1e6:.1f}M | Klassen: {len(self.classes)}",
        )

    # ── 4. Training ─────────────────────────────────────────────────────────
    def train(self) -> None:
        import torch
        from PIL import Image
        from transformers import Trainer, TrainerCallback

        self.device_used = hft.device_name()
        MessageProtocol.status("training", "Training gestartet...")
        MessageProtocol.status("training", f"Gerät: {self.device_used.upper()}")

        processor = self.processor

        def collate(batch):
            images = []
            for row in batch:
                with Image.open(row["path"]) as im:
                    images.append(im.convert("RGB"))
            enc = processor(images=images, return_tensors="pt")
            enc["labels"] = torch.tensor([row["labels"] for row in batch], dtype=torch.long)
            return enc

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return hft.classification_scores(list(labels), list(preds))

        total_steps = max(
            (len(self.train_dataset) // max(self.config.batch_size, 1))
            * max(self.config.epochs, 1), 1)
        if int(self.config.max_steps) > 0:
            total_steps = int(self.config.max_steps)

        args = hft.build_training_arguments(
            self.config,
            self.config.effective_output_dir(),
            __import__("transformers").TrainingArguments,
            remove_unused_columns=False,   # 'path' wird im Collator gebraucht
        )
        self._trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=collate,
            compute_metrics=compute_metrics,
            callbacks=[hft.progress_callback(TrainerCallback, self, total_steps)],
        )
        self._start_time = time.time()
        self._trainer.train()
        MessageProtocol.status("training", "Training abgeschlossen")

    # ── 5. Validierung ──────────────────────────────────────────────────────
    def validate(self) -> Dict[str, float]:
        MessageProtocol.status("validating", "Finale Validierung...")
        result = self._trainer.evaluate()
        return hft.final_metrics(
            self, self._trainer, result, self._start_time,
            architecture=self.model_type, num_labels=len(self.classes),
        )

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.processor.save_pretrained(str(out))
        import json
        (out / "label_mapping.json").write_text(
            json.dumps({"classes": self.classes}, indent=2, ensure_ascii=False), encoding="utf-8")
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)
