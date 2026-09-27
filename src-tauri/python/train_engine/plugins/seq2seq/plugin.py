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

from ft_data.seq2seq import (
    batch_texts, describe, file_split_name, generation_scores, resolve_spec, save_spec,
)


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.model_type = ""
        self.device_used = "cpu"
        self.spec: Dict[str, Any] = {}
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
            loose = [f for f in sorted(root.rglob("*")) if f.suffix.lower() in exts
                     and ".frametrain_media" not in f.parts]
            if not loose:
                raise ValueError(f"Keine Datendateien in '{root}' gefunden (erwartet: {', '.join(exts)}).")
            # train.csv / validation-00000-of-00001.parquet / test.jsonl im Root nach
            # Namen zuordnen. Vorher landete alles im Training — auch die Testdaten.
            by_split: Dict[str, List[str]] = {}
            for f in loose:
                split = file_split_name(f.stem) or "train"
                by_split.setdefault({"val": "validation"}.get(split, split), []).append(str(f))
            if "train" not in by_split:
                by_split["train"] = by_split.pop(next(iter(by_split)))
            data_files = by_split

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
        first_row = train_raw[0] if len(train_raw) else {}
        self.spec = resolve_spec(cols, first_row, self.config.plugin_config or {})
        MessageProtocol.status("loading_data", f"Eingabe → Ziel: {describe(self.spec)}")

        eval_raw = raw.get("validation") or raw.get("test")
        if eval_raw is None:
            split = train_raw.train_test_split(test_size=0.1, seed=self.config.seed)
            train_raw, eval_raw = split["train"], split["test"]
            MessageProtocol.status("loading_data", "Kein Validierungs-Split gefunden — 10% abgetrennt.")

        eval_raw = hft.cap_eval_dataset(eval_raw, getattr(self.config, "max_eval_samples", 0), self.config.seed)
        # Rohtexte fuer ROUGE/BLEU am Ende — tokenisiert laesst sich das Ziel nicht vergleichen.
        self._eval_raw = eval_raw

        tokenizer = self.tokenizer
        spec, prefix = self.spec, self.prefix
        max_src, max_tgt = self.config.max_seq_length, self.max_target_length

        def tokenize(batch):
            # Dict-Spalten (Uebersetzung) und Listen-Ziele (keyphrases) werden hier zu Text.
            sources, targets = batch_texts(batch, spec)
            inputs = [f"{prefix}{t}" for t in sources]
            enc = tokenizer(inputs, truncation=True, max_length=max_src)
            labels = tokenizer(text_target=targets, truncation=True, max_length=max_tgt)
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
        metrics.update(self._generation_metrics())
        return metrics

    def _generation_metrics(self) -> Dict[str, float]:
        """ROUGE-1/2/L und BLEU ueber echte Generierung auf dem Val-Split.

        Der Val-Loss sagt wenig darueber, ob die Zusammenfassung oder Uebersetzung
        taugt. Gedeckelt auf Max Eval Samples bzw. 200 Beispiele — Generieren ist
        um ein Vielfaches langsamer als ein Loss-Durchlauf.
        """
        raw = getattr(self, "_eval_raw", None)
        if raw is None or len(raw) == 0 or self.is_stopped:
            return {}
        import torch

        cap = int(getattr(self.config, "max_eval_samples", 0) or 0) or 200
        if len(raw) > cap:
            raw = raw.shuffle(seed=self.config.seed).select(range(cap))
        sources, targets = batch_texts(raw[:], self.spec)
        MessageProtocol.status("evaluating", f"ROUGE/BLEU: erzeuge {len(sources)} Texte aus dem Val-Split...")
        model, tok = self.model, self.tokenizer
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        preds: List[str] = []
        bs = max(1, int(self.config.batch_size))
        try:
            for i in range(0, len(sources), bs):
                if self.is_stopped:
                    return {}
                enc = tok([f"{self.prefix}{t}" for t in sources[i:i + bs]], return_tensors="pt",
                          padding=True, truncation=True, max_length=self.config.max_seq_length).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=self.max_target_length, num_beams=1)
                preds.extend(tok.batch_decode(out, skip_special_tokens=True))
        except Exception as exc:  # Metrik-Zugabe — ein Fehler hier darf das Modell nicht kosten
            MessageProtocol.warning(f"ROUGE/BLEU nicht berechnet: {type(exc).__name__}: {exc}")
            return {}
        finally:
            if was_training:
                model.train()
        scores, notes = generation_scores(preds, targets)
        for note in notes:
            MessageProtocol.warning(note)
        if scores:
            MessageProtocol.status(
                "evaluating",
                " | ".join(f"{k} {v:.3f}" if k != "bleu" else f"BLEU {v:.1f}" for k, v in scores.items()))
        return scores

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        # Spalten und Prefix neben dem Modell, damit der Test dieselben nutzt.
        save_spec(out, self.spec, self.prefix)
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)
