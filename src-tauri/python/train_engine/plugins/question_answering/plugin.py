"""
Extraktives Frage-Antwort (SQuAD-Stil)
======================================
AutoModelForQuestionAnswering auf einem Encoder (BERT, RoBERTa, DeBERTa,
XLM-R, DistilBERT …). Ein Basis-Encoder bekommt einen frischen Start/End-Kopf,
ein schon trainiertes QA-Modell (deepset/roberta-base-squad2) wird
weitertrainiert.

Daten und Bewertung kommen aus ft_data.qa (gemeinsam mit dem Test-Plugin):
SQuAD-JSON, HF-SQuAD-Zeilen oder flache Tabellen context/question/answer.
Lange Kontexte werden in Fenster geteilt (doc_stride Tokens Ueberlappung).

Gemessen wird auf dem Val-Split Exact Match und F1 wie beim SQuAD-Skript —
einmal vor und einmal nach dem Training.

plugin_config:
    context_column, question_column, answer_column, answer_start_column
    doc_stride          Ueberlappung der Fenster in Tokens (Standard 128)
    max_answer_length   laengste Antwort in Tokens (Standard 30)
    allow_no_answer     "auto" (an, sobald das Dataset Fragen ohne Antwort hat) | true | false
    null_threshold      Abstand, ab dem "keine Antwort" gewinnt (Standard 0)
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft

from ft_data.splits import split_files
from ft_data import qa as Q

SUPPORTED_ARCHITECTURES = {
    "bert", "distilbert", "roberta", "xlm-roberta", "deberta", "deberta-v2",
    "electra", "albert", "camembert", "mpnet", "modernbert", "mobilebert", "longformer", "splinter",
}


def _auto_flag(value: Any, default: bool) -> bool:
    s = str(value if value is not None else "auto").strip().lower()
    if s in ("", "auto"):
        return default
    return s in ("1", "true", "yes", "ja", "on")


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        pc = config.plugin_config or {}
        self.pc = pc
        self.model_type = ""
        self.tokenizer = None
        self.model = None
        self.device_used = "cpu"
        self.train_examples: List[Dict[str, Any]] = []
        self.eval_examples: List[Dict[str, Any]] = []
        self.train_dataset = None
        self.eval_dataset = None
        self.eval_features: Dict[str, List[Any]] = {}
        self.max_length = 384
        self.doc_stride = int(pc.get("doc_stride", 128) or 128)
        self.max_answer_length = int(pc.get("max_answer_length", 30) or 30)
        self.null_threshold = float(pc.get("null_threshold", 0.0) or 0.0)
        self.allow_no_answer = False
        self.baseline: Dict[str, float] = {}
        self._trainer = None
        self._start_time = time.time()
        self._last_train_loss = 0.0
        self._last_lr = config.learning_rate

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        import json
        from transformers import AutoTokenizer

        root = Path(self.config.model_path)
        cfg_path = root / "config.json"
        if cfg_path.exists():
            try:
                self.model_type = str(json.loads(cfg_path.read_text(encoding="utf-8")).get("model_type", ""))
            except Exception:
                self.model_type = ""
        if self.model_type and self.model_type not in SUPPORTED_ARCHITECTURES:
            MessageProtocol.warning(
                f"Architektur '{self.model_type}' ist fuer Frage-Antwort nicht geprueft — "
                "es wird trotzdem versucht (AutoModelForQuestionAnswering).")
        self.tokenizer = AutoTokenizer.from_pretrained(str(root))
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError(
                "Frage-Antwort braucht einen 'schnellen' Tokenizer (Offsets der Zeichen), dieses Modell "
                "bringt nur einen langsamen mit. Bitte ein Modell mit tokenizer.json waehlen.")
        model_max = int(getattr(self.tokenizer, "model_max_length", 512) or 512)
        if model_max > 100_000:
            model_max = 512
        wanted = int(self.config.max_seq_length or 0)
        # 128 ist der allgemeine Formular-Standard und fuer Kontexte zu kurz.
        self.max_length = min(model_max, wanted if wanted > 128 else 384)
        self.doc_stride = max(16, min(self.doc_stride, self.max_length // 2))
        MessageProtocol.status(
            "init", f"Architektur: {self.model_type or 'unbekannt'} | Fenster: {self.max_length} Tokens, "
                    f"Ueberlappung {self.doc_stride}")

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def _load(self, files) -> List[Dict[str, Any]]:
        rows, stats = Q.normalize_rows(Q.load_files(files), self.pc)
        MessageProtocol.status("loading_data", Q.describe(stats))
        return rows

    def load_data(self) -> None:
        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        files = split_files(root, Q.DATA_EXTS)
        if not files.get("train"):
            raise ValueError(f"Keine Datendateien in '{root}' gefunden (erwartet: {', '.join(Q.DATA_EXTS)}).")
        train = self._load(files["train"])
        if not train:
            raise ValueError("Keine verwertbare Frage gefunden (Kontext, Frage oder Antwort fehlen).")
        if files.get("val"):
            val = self._load(files["val"])
        else:
            import random
            rng = random.Random(self.config.seed)
            rng.shuffle(train)
            n_val = max(1, int(round(len(train) * 0.1))) if len(train) >= 10 else 0
            if n_val:
                val, train = train[:n_val], train[n_val:]
                MessageProtocol.status("loading_data", f"Kein Validierungs-Split gefunden — {n_val} Fragen abgetrennt.")
            else:
                val = list(train)
                MessageProtocol.status("loading_data", "Sehr kleines Dataset — Validierung auf den Trainingsdaten (nur Lernnachweis).")
        cap = int(getattr(self.config, "max_eval_samples", 0) or 0)
        if cap and len(val) > cap:
            import random
            val = random.Random(self.config.seed).sample(val, cap)
            MessageProtocol.status("loading_data", f"Val-Split auf {cap} Fragen begrenzt (Max Eval Samples).")
        has_null = any(not e["answers"]["text"] for e in train + val)
        self.allow_no_answer = _auto_flag(self.pc.get("allow_no_answer"), has_null)
        self.train_examples, self.eval_examples = train, val

        from datasets import Dataset
        tr = Q.prepare_train_features(self.tokenizer, train, self.max_length, self.doc_stride)
        ev = Q.prepare_train_features(self.tokenizer, val, self.max_length, self.doc_stride)
        self.train_dataset = Dataset.from_dict(tr)
        self.eval_dataset = Dataset.from_dict(ev)
        self.eval_features = Q.prepare_eval_features(self.tokenizer, val, self.max_length, self.doc_stride)
        MessageProtocol.status(
            "loading_data",
            f"✓ Train: {len(train)} Fragen in {len(self.train_dataset)} Fenstern | Val: {len(val)} Fragen"
            + (" | 'keine Antwort' erlaubt (SQuAD v2)" if self.allow_no_answer else ""))

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        from transformers import AutoModelForQuestionAnswering

        self.device_used = hft.device_name()
        MessageProtocol.status("building_model", "Lade Frage-Antwort-Modell...")
        self.model = AutoModelForQuestionAnswering.from_pretrained(str(self.config.model_path))
        self.model.to(self.device_used)
        params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status("building_model", f"✓ Modell geladen | {params/1e6:.1f}M Parameter | Gerät: {self.device_used.upper()}")
        self.baseline = self.evaluate_span_metrics("vor dem Training")

    def predict(self, examples: List[Dict[str, Any]], features: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
        """Logits je Fenster -> Antwort je Frage (ft_data.qa.postprocess)."""
        import torch
        model = self.model
        was_training = model.training
        model.eval()
        keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in features]
        bs = max(int(self.config.batch_size), 8)
        starts, ends = [], []
        with torch.no_grad():
            for i in range(0, len(features["input_ids"]), bs):
                batch = {k: torch.tensor(features[k][i:i + bs]).to(model.device) for k in keys}
                out = model(**batch)
                starts.extend(out.start_logits.float().cpu().numpy())
                ends.extend(out.end_logits.float().cpu().numpy())
        if was_training:
            model.train()
        return Q.postprocess(examples, features, starts, ends, max_answer_length=self.max_answer_length,
                             allow_no_answer=self.allow_no_answer, null_threshold=self.null_threshold)

    def evaluate_span_metrics(self, when: str) -> Dict[str, float]:
        if not self.eval_examples:
            return {}
        preds = self.predict(self.eval_examples, self.eval_features)
        m = Q.squad_metrics({k: v["text"] for k, v in preds.items()}, self.eval_examples)
        if m:
            MessageProtocol.status("evaluating", f"{when}: Exact Match {m['exact_match']:.1%}, F1 {m['f1']:.1%}")
        return m

    # ── 4. Training ─────────────────────────────────────────────────────────
    def train(self) -> None:
        from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator

        total = max(1, (len(self.train_dataset) // max(int(self.config.batch_size), 1))) * max(1, int(self.config.epochs))
        if int(self.config.max_steps) > 0:
            total = int(self.config.max_steps)
        overrides: Dict[str, Any] = {"label_smoothing_factor": 0.0}
        if not len(self.eval_dataset):
            overrides["eval_strategy"] = "no"
        args = hft.build_training_arguments(self.config, self.config.effective_output_dir(), TrainingArguments, **overrides)
        self._trainer = Trainer(
            model=self.model, args=args, train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset if len(self.eval_dataset) else None,
            data_collator=default_data_collator,
            callbacks=[hft.progress_callback(TrainerCallback, self, total)],
        )
        MessageProtocol.status("training", f"Training gestartet | Gerät: {self.device_used.upper()}")
        self._start_time = time.time()
        self._trainer.train(resume_from_checkpoint=hft.resume_checkpoint(self.config))
        MessageProtocol.status("training", "Training abgeschlossen")

    # ── 5. Validierung ──────────────────────────────────────────────────────
    def validate(self) -> Dict[str, Any]:
        MessageProtocol.status("validating", "Finale Validierung...")
        result = self._trainer.evaluate() if self._trainer and self._trainer.eval_dataset is not None else {}
        metrics = hft.final_metrics(self, self._trainer, result, self._start_time,
                                    architecture=self.model_type, num_labels=0)
        for key in ("accuracy", "f1", "precision", "recall", "num_labels"):
            metrics.pop(key, None)
        now = self.evaluate_span_metrics("nach dem Training")
        metrics.update(now)
        for k, v in self.baseline.items():
            metrics[f"{k}_before"] = v
        if "exact_match" in now:
            # Die Analyse-Seite zeigt "accuracy" prominent — bei QA ist das Exact Match.
            metrics["accuracy"] = now["exact_match"]
        metrics["n_train_questions"] = len(self.train_examples)
        metrics["n_val_questions"] = len(self.eval_examples)
        metrics["allow_no_answer"] = bool(self.allow_no_answer)
        if now and self.baseline:
            MessageProtocol.status(
                "validating",
                f"Vorher -> nachher: EM {self.baseline['exact_match']:.1%} -> {now['exact_match']:.1%}, "
                f"F1 {self.baseline['f1']:.1%} -> {now['f1']:.1%}")
        return metrics

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out), safe_serialization=True)
        self.tokenizer.save_pretrained(str(out))
        Q.save_spec(out, {
            "task": "question_answering", "max_length": self.max_length, "doc_stride": self.doc_stride,
            "max_answer_length": self.max_answer_length, "allow_no_answer": bool(self.allow_no_answer),
            "null_threshold": self.null_threshold,
        })
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)
