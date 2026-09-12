"""
Audio Classification (HuggingFace)
==================================
Schliesst die Luecke "Sprache/Audio geht gar nicht".

Trainiert Audiomodelle auf Klassifikation: Sprecher, Kommandos, Stimmungen,
Geraeusche. Kein ASR (Transkription) — dafuer braucht es CTC bzw. Seq2Seq,
das ist ein eigener Aufgabenbereich.

Dataset-Layout: ein Ordner pro Klasse, optional in train/ val/ test/.
    <dataset>/train/ja/*.wav
    <dataset>/val/nein/*.wav
HuggingFace-Parquet mit Bild-/Audio-Spalte und Label wird einmalig in
Klassenordner entpackt (ft_data.media).
"""
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft
from ft_data.media import resolve_class_layout


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.extractor = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.classes: List[str] = []
        self.model_type = ""
        self.device_used = "cpu"
        self.sampling_rate = 16000
        self.max_seconds = float(config.get_plugin_value("max_seconds", 10.0))
        self._trainer = None
        self._start_time = time.time()
        self._last_train_loss = 0.0
        self._last_lr = config.learning_rate

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        import json
        from transformers import AutoFeatureExtractor

        cfg_path = Path(self.config.model_path) / "config.json"
        if cfg_path.exists():
            try:
                self.model_type = json.loads(cfg_path.read_text(encoding="utf-8")).get("model_type", "")
            except Exception:
                self.model_type = ""
        MessageProtocol.status("init", f"✓ Architektur erkannt: {self.model_type or 'unbekannt'} | Lade Feature-Extractor...")
        self.extractor = AutoFeatureExtractor.from_pretrained(self.config.model_path)
        self.sampling_rate = int(getattr(self.extractor, "sampling_rate", 16000) or 16000)
        MessageProtocol.status("init", f"Feature-Extractor geladen ✓ | Abtastrate: {self.sampling_rate} Hz")

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def load_data(self) -> None:
        from datasets import Dataset

        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")

        # Gemeinsame Regeln fuer Klassenordner, Splits und HF-Parquet (ft_data).
        # Validierung ist val/; fehlt sie, wird von train abgetrennt — test/
        # bleibt fuer den Test unberuehrt.
        layout = resolve_class_layout(
            root, "audio", seed=self.config.seed,
            status=lambda m: MessageProtocol.status("loading_data", m))
        for note in layout.notes:
            MessageProtocol.status("loading_data", note)
        self.classes = layout.classes

        def as_dict(items):
            return {"path": [str(f) for f, _ in items], "labels": [label for _, label in items]}

        MessageProtocol.status(
            "loading_data",
            f"Klassen ({len(self.classes)}): {', '.join(self.classes)} | "
            f"{len(layout.train)} Trainingsdateien",
        )
        train_ds = Dataset.from_dict(as_dict(layout.train))
        eval_ds = Dataset.from_dict(as_dict(layout.val))

        eval_ds = hft.cap_eval_dataset(eval_ds, getattr(self.config, "max_eval_samples", 0), self.config.seed)
        self.train_dataset = train_ds
        self.eval_dataset = eval_ds
        MessageProtocol.status("loading_data", f"✓ Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        from transformers import AutoModelForAudioClassification

        MessageProtocol.status("building_model", "Lade Modell fuer Audioklassifikation...")
        self.model = AutoModelForAudioClassification.from_pretrained(
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

        extractor = self.extractor
        target_sr = self.sampling_rate
        max_len = int(self.max_seconds * target_sr)

        def load_wave(path: str) -> np.ndarray:
            import librosa
            wave, _ = librosa.load(path, sr=target_sr, mono=True)
            if len(wave) > max_len:
                wave = wave[:max_len]
            return wave

        def collate(batch):
            waves = [load_wave(row["path"]) for row in batch]
            enc = extractor(waves, sampling_rate=target_sr, return_tensors="pt", padding=True)
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
        self.extractor.save_pretrained(str(out))
        import json
        (out / "label_mapping.json").write_text(
            json.dumps({"classes": self.classes, "sampling_rate": self.sampling_rate},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)
