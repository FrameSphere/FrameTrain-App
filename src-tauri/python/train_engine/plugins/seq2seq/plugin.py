"""
Seq2Seq (HuggingFace)
=====================
Encoder-Decoder-Modelle: Zusammenfassung, Übersetzung, Textumformung.

Erwartet zwei Spalten — Eingabetext und Zieltext. Erkannt werden gängige
Namenspaare; wer andere benutzt, setzt sie in plugin_config:
    {"source_column": "artikel", "target_column": "kurzfassung"}
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft

SOURCE_CANDIDATES = ["source", "input", "text", "article", "document", "de", "src", "question"]
TARGET_CANDIDATES = ["target", "output", "summary", "highlights", "translation", "en", "tgt", "answer"]


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.model_type = ""
        self.device_used = "cpu"
        self.source_col = config.get_plugin_value("source_column")
        self.target_col = config.get_plugin_value("target_column")
        self.prefix = str(config.get_plugin_value("task_prefix", "") or "")
        self.max_target_length = int(config.get_plugin_value("max_target_length", 128))
        self._trainer = None
        self._start_time = time.time()
        self._last_train_loss = 0.0
        self._last_lr = config.learning_rate

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        from transformers import AutoTokenizer

        cfg_path = Path(self.config.model_path) / "config.json"
        if cfg_path.exists():
            try:
                self.model_type = json.loads(cfg_path.read_text(encoding="utf-8")).get("model_type", "")
            except Exception:
                self.model_type = ""
        MessageProtocol.status("init", f"✓ Architektur erkannt: {self.model_type or 'unbekannt'} | Lade Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        MessageProtocol.status("init", "Tokenizer geladen ✓")

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def _load_files(self):
        from datasets import load_dataset

        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")

        exts = (".json", ".jsonl", ".csv", ".tsv", ".parquet")

        def files_in(sub: str) -> List[str]:
            d = root / sub
            return [str(f) for f in sorted(d.rglob("*")) if f.suffix.lower() in exts] if d.is_dir() else []

        data_files: Dict[str, List[str]] = {}
        for split, subs in (("train", ["train"]), ("validation", ["val", "validation"]), ("test", ["test"])):
            for sub in subs:
                found = files_in(sub)
                if found:
                    data_files[split] = found
                    break
        if not data_files:
            loose = [str(f) for f in sorted(root.rglob("*")) if f.suffix.lower() in exts]
            if not loose:
                raise ValueError(f"Keine Datendateien in '{root}' gefunden (erwartet: {', '.join(exts)}).")
            data_files["train"] = loose

        ext = Path(data_files["train"][0]).suffix.lower()
        if ext in (".json", ".jsonl"):
            return load_dataset("json", data_files=data_files)
        if ext == ".parquet":
            return load_dataset("parquet", data_files=data_files)
        if ext == ".tsv":
            return load_dataset("csv", data_files=data_files, delimiter="\t")
        return load_dataset("csv", data_files=data_files)

    def load_data(self) -> None:
        raw = self._load_files()
        train_raw = raw["train"]
        cols = list(train_raw.features.keys())

        if not self.source_col:
            self.source_col = next((c for c in SOURCE_CANDIDATES if c in cols), None)
        if not self.target_col:
            self.target_col = next((c for c in TARGET_CANDIDATES if c in cols and c != self.source_col), None)
        if not self.source_col or not self.target_col:
            raise ValueError(
                f"Eingabe- und Zielspalte nicht erkannt. Vorhandene Spalten: {cols}.\n"
                "Setze sie in der Plugin-Konfiguration, z.B.:\n"
                '  {"source_column": "artikel", "target_column": "kurzfassung"}'
            )
        MessageProtocol.status(
            "loading_data",
            f"Eingabe-Spalte: '{self.source_col}' → Ziel-Spalte: '{self.target_col}'",
        )

        eval_raw = raw.get("validation") or raw.get("test")
        if eval_raw is None:
            split = train_raw.train_test_split(test_size=0.1, seed=self.config.seed)
            train_raw, eval_raw = split["train"], split["test"]
            MessageProtocol.status("loading_data", "Kein Validierungs-Split gefunden — 10% abgetrennt.")

        eval_raw = hft.cap_eval_dataset(eval_raw, getattr(self.config, "max_eval_samples", 0), self.config.seed)

        tokenizer = self.tokenizer
        src, tgt, prefix = self.source_col, self.target_col, self.prefix
        max_src, max_tgt = self.config.max_seq_length, self.max_target_length

        def tokenize(batch):
            inputs = [f"{prefix}{t}" for t in batch[src]]
            enc = tokenizer(inputs, truncation=True, max_length=max_src)
            labels = tokenizer(text_target=batch[tgt], truncation=True, max_length=max_tgt)
            enc["labels"] = labels["input_ids"]
            return enc

        self.train_dataset = train_raw.map(tokenize, batched=True, remove_columns=train_raw.column_names)
        self.eval_dataset = eval_raw.map(tokenize, batched=True, remove_columns=eval_raw.column_names)
        MessageProtocol.status(
            "loading_data",
            f"✓ Dataset tokenisiert | Train: {len(self.train_dataset)} | Eval: {len(self.eval_dataset)}",
        )

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        from transformers import AutoModelForSeq2SeqLM

        MessageProtocol.status("building_model", "Lade Seq2Seq-Modell...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
        params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status("building_model", f"✓ Modell geladen | Parameter: {params/1e6:.1f}M")

    # ── 4. Training ─────────────────────────────────────────────────────────
    def train(self) -> None:
        from transformers import (
            DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback,
        )

        self.device_used = hft.device_name()
        MessageProtocol.status("training", "Training gestartet...")
        MessageProtocol.status("training", f"Gerät: {self.device_used.upper()}")

        total_steps = max(
            (len(self.train_dataset) // max(self.config.batch_size, 1))
            * max(self.config.epochs, 1), 1)
        if int(self.config.max_steps) > 0:
            total_steps = int(self.config.max_steps)

        args = hft.build_training_arguments(
            self.config, self.config.effective_output_dir(), Seq2SeqTrainingArguments,
        )
        self._trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model),
            callbacks=[hft.progress_callback(TrainerCallback, self, total_steps)],
        )
        self._start_time = time.time()
        self._trainer.train()
        MessageProtocol.status("training", "Training abgeschlossen")

    # ── 5. Validierung ──────────────────────────────────────────────────────
    def validate(self) -> Dict[str, float]:
        MessageProtocol.status("validating", "Finale Validierung...")
        result = self._trainer.evaluate()
        metrics = hft.final_metrics(
            self, self._trainer, result, self._start_time,
            architecture=self.model_type, num_labels=0,
        )
        # Seq2Seq hat keine Klassen — die Kennzahlen wären sonst irrefuehrende Nullen.
        for key in ("accuracy", "f1", "precision", "recall", "num_labels"):
            metrics.pop(key, None)
        return metrics

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)
