"""
Video Classification (HuggingFace)
==================================
Trainiert Videomodelle auf Klassifikation von Clips: Bewegungen, Handlungen,
Ablaeufe ("springt", "faellt", "steht").

Dataset-Layout: ein Ordner pro Klasse, optional in train/ val/ test/.
    <dataset>/train/springt/*.mp4
    <dataset>/val/faellt/*.mp4
So exportiert die Datensatz-Werkstatt Videoprojekte.

Je Clip sieht das Modell num_frames Einzelbilder (aus seiner config.json,
bei VideoMAE 16), gleichmaessig ueber den Clip verteilt. Die Auswahl steckt
in ft_data.video und ist in Training, Test und Werkstatt dieselbe.
"""
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft
from ft_data.media import resolve_class_layout
from ft_data.video import read_clip_frames


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.processor = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.classes: List[str] = []
        self.model_type = ""
        self.num_frames = int(config.get_plugin_value("num_frames", 0) or 0)
        self.device_used = "cpu"
        self._trainer = None
        self._start_time = time.time()
        self._last_train_loss = 0.0
        self._last_lr = config.learning_rate

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        import json
        from transformers import AutoImageProcessor

        try:
            import cv2  # noqa: F401
        except ImportError:
            raise ImportError("OpenCV fehlt. Installiere: pip install opencv-python")

        cfg_path = Path(self.config.model_path) / "config.json"
        cfg = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
        self.model_type = cfg.get("model_type", "")
        # Die Bildzahl gehoert zum Modell: VideoMAE ist auf 16 trainiert, ein
        # anderer Wert passt nicht zu seinen Positions-Embeddings.
        if self.num_frames <= 0:
            self.num_frames = int(cfg.get("num_frames", 16) or 16)
        MessageProtocol.status("init", f"✓ Architektur erkannt: {self.model_type or 'unbekannt'} | "
                                       f"{self.num_frames} Bilder je Clip | Lade Bild-Prozessor...")
        self.processor = AutoImageProcessor.from_pretrained(self.config.model_path)
        MessageProtocol.status("init", "Bild-Prozessor geladen ✓")

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def load_data(self) -> None:
        from datasets import Dataset

        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")

        layout = resolve_class_layout(
            root, "video", seed=self.config.seed,
            status=lambda m: MessageProtocol.status("loading_data", m))
        for note in layout.notes:
            MessageProtocol.status("loading_data", note)
        self.classes = layout.classes

        def as_dict(items):
            return {"path": [str(f) for f, _ in items], "labels": [label for _, label in items]}

        MessageProtocol.status(
            "loading_data",
            f"Klassen ({len(self.classes)}): {', '.join(self.classes)} | {len(layout.train)} Trainingsclips",
        )
        train_ds = Dataset.from_dict(as_dict(layout.train))
        eval_ds = Dataset.from_dict(as_dict(layout.val))
        eval_ds = hft.cap_eval_dataset(eval_ds, getattr(self.config, "max_eval_samples", 0), self.config.seed)
        self.train_dataset = train_ds
        self.eval_dataset = eval_ds
        MessageProtocol.status("loading_data", f"✓ Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        from transformers import AutoModelForVideoClassification

        MessageProtocol.status("building_model", "Lade Modell fuer Videoklassifikation...")
        self.model = AutoModelForVideoClassification.from_pretrained(
            self.config.model_path,
            num_labels=len(self.classes),
            id2label={i: c for i, c in enumerate(self.classes)},
            label2id={c: i for i, c in enumerate(self.classes)},
            ignore_mismatched_sizes=True,
        )
        params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status(
            "building_model",
            f"✓ Modell geladen | Parameter: {params/1e6:.1f}M | Klassen: {len(self.classes)}",
        )

    # ── 4. Training ─────────────────────────────────────────────────────────
    def train(self) -> None:
        import torch
        from transformers import Trainer, TrainerCallback

        self.device_used = hft.device_name()
        MessageProtocol.status("training", "Training gestartet...")
        MessageProtocol.status("training", f"Gerät: {self.device_used.upper()}")

        processor = self.processor
        num_frames = self.num_frames

        def collate(batch):
            clips = [read_clip_frames(row["path"], num_frames) for row in batch]
            enc = processor(clips, return_tensors="pt")
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
            remove_unused_columns=False,
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
            json.dumps({"classes": self.classes, "num_frames": self.num_frames},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)
