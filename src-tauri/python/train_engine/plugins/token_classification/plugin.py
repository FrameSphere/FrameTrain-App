"""
Token Classification (HuggingFace)
==================================
Ein Label pro Wort statt pro Text: Named Entity Recognition ("Karol" = PER,
"Berlin" = LOC), Wortarten (POS) und aehnliche Aufgaben. Jeder Encoder, den
AutoModelForTokenClassification laden kann (BERT, DistilBERT, RoBERTa,
XLM-RoBERTa, DeBERTa, ELECTRA, ALBERT, ...).

Datenformate (Erkennung in ft_data.tokens):
    JSONL/JSON/Parquet   {"tokens": [...], "ner_tags": [...]}   (Tags als Text oder ints)
    CoNLL                "Karol B-PER" pro Zeile, Leerzeile = Satzgrenze
    Spans                {"text": "...", "entities": [{"start", "end", "label"}]}
Split: train/ val/ test/ Ordner oder Dateinamen, sonst 10 % Validierung.

plugin_config:
    tokens_column / tags_column   eigene Spaltennamen
    label_list                    Namen fuer int-Tags ohne ClassLabel-Info
    label_all_tokens              auch Folge-Subtokens labeln (Standard: nein)
"""
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft

from ft_data.splits import split_files
from ft_data import tokens as tk

# BPE-Tokenizer (RoBERTa & Co.) brauchen fuer schon zerlegte Woerter ein
# Leerzeichen davor — sonst wirft der Tokenizer bei is_split_into_words.
PREFIX_SPACE_TYPES = {"roberta", "longformer", "deberta", "bart", "gpt2", "mvp", "led"}


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.model_type = ""
        self.labels: List[str] = []
        self.spec: Dict = {}
        self.device_used = "cpu"
        self.label_all_tokens = bool(config.get_plugin_value("label_all_tokens", False))
        self._trainer = None
        self._start_time = time.time()
        self._last_train_loss = 0.0
        self._last_lr = config.learning_rate
        self._scheme = "entity"

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
        kwargs = {"add_prefix_space": True} if self.model_type in PREFIX_SPACE_TYPES else {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, **kwargs)
        if not getattr(self.tokenizer, "is_fast", False):
            # word_ids() gibt es nur bei den Rust-Tokenizern. Ohne sie laesst
            # sich nicht sagen, welches Subtoken zu welchem Wort gehoert.
            MessageProtocol.error(
                "Tokenizer ohne Wortzuordnung",
                f"Fuer {self.model_type or 'dieses Modell'} gibt es nur einen langsamen Tokenizer. "
                "Token-Klassifikation braucht einen 'Fast'-Tokenizer (tokenizer.json).\n"
                + __import__("ft_data.deps", fromlist=["install_hint"]).install_hint("tokenizers", "sentencepiece", "protobuf"),
            )
            return False
        MessageProtocol.status("init", "Tokenizer geladen ✓")

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def load_data(self) -> None:
        from datasets import Dataset

        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        files = split_files(root, tk.DATA_EXTS)
        if not files.get("train"):
            raise ValueError(
                f"Keine Daten in '{root}' gefunden. Erwartet: {', '.join(tk.DATA_EXTS)} "
                "(JSONL mit tokens + ner_tags, CoNLL-Text oder text + entities)."
            )
        cfg = self.config.plugin_config or {}
        train_rows, self.spec = tk.load_files(files["train"], cfg)
        if not train_rows:
            raise ValueError("Keine verwertbaren Saetze gefunden (Token- und Tag-Liste gleich lang?).")
        fmt = {"tokens": "Token-Listen", "conll": "CoNLL", "spans": "Text mit Spans"}.get(self.spec.get("format"), "?")
        MessageProtocol.status("loading_data", f"Format: {fmt} | {len(train_rows)} Saetze im Training")

        if files.get("val"):
            eval_rows, _ = tk.load_files(files["val"], cfg)
        else:
            import random
            rng = random.Random(self.config.seed)
            order = list(range(len(train_rows)))
            rng.shuffle(order)
            n_val = max(1, int(round(len(train_rows) * 0.1))) if len(train_rows) > 1 else 0
            eval_rows = [train_rows[i] for i in order[:n_val]]
            train_rows = [train_rows[i] for i in order[n_val:]]
            MessageProtocol.status("loading_data", "Kein Validierungs-Split gefunden — 10% abgetrennt.")

        self.labels = tk.build_label_list(train_rows + eval_rows)
        self._scheme = "entity" if tk.is_bio_scheme(self.labels) else "token"
        MessageProtocol.status(
            "loading_data",
            f"Labels ({len(self.labels)}): {', '.join(self.labels[:30])}"
            + (" …" if len(self.labels) > 30 else "")
            + (" | Metrik: Entitaeten (seqeval)" if self._scheme == "entity" else " | Metrik: pro Token"),
        )
        label2id = {l: i for i, l in enumerate(self.labels)}
        # Folge-Subtokens eines B-X bekommen I-X, falls label_all_tokens an ist.
        b_to_i = {label2id[l]: label2id.get("I-" + l[2:], label2id[l])
                  for l in self.labels if l.startswith("B-")}

        tokenizer = self.tokenizer
        max_len = int(self.config.max_seq_length or 128)
        label_all = self.label_all_tokens

        def encode(batch):
            enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
            enc["labels"] = [
                tk.align_labels(enc.word_ids(i), [label2id[t] for t in tags], label_all, b_to_i)
                for i, tags in enumerate(batch["tags"])
            ]
            return enc

        def as_ds(rows):
            return Dataset.from_dict({"tokens": [r["tokens"] for r in rows], "tags": [r["tags"] for r in rows]})

        train_ds = as_ds(train_rows)
        eval_ds = hft.cap_eval_dataset(as_ds(eval_rows), getattr(self.config, "max_eval_samples", 0), self.config.seed)
        self.train_dataset = train_ds.map(encode, batched=True, remove_columns=train_ds.column_names)
        self.eval_dataset = eval_ds.map(encode, batched=True, remove_columns=eval_ds.column_names)
        MessageProtocol.status(
            "loading_data",
            f"✓ Dataset tokenisiert | Train: {len(self.train_dataset)} | Eval: {len(self.eval_dataset)}",
        )

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        from transformers import AutoModelForTokenClassification

        MessageProtocol.status("building_model", "Lade Modell fuer Token-Klassifikation...")
        # ignore_mismatched_sizes: ein fertiges NER-Modell mit anderen Labels
        # (z.B. 9 CoNLL-Tags) laesst sich so auf eigene Labels umlernen.
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_path,
            num_labels=len(self.labels),
            id2label={i: l for i, l in enumerate(self.labels)},
            label2id={l: i for i, l in enumerate(self.labels)},
            ignore_mismatched_sizes=True,
        )
        params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status(
            "building_model",
            f"✓ Modell geladen | Parameter: {params/1e6:.1f}M | Labels: {len(self.labels)}",
        )

    # ── 4. Training ─────────────────────────────────────────────────────────
    def _compute_metrics(self, eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_tags, pred_tags = [], []
        for p_row, l_row in zip(preds, label_ids):
            t_seq, p_seq = [], []
            for p, l in zip(p_row, l_row):
                if l == -100:
                    continue
                t_seq.append(self.labels[int(l)])
                p_seq.append(self.labels[int(p)])
            true_tags.append(t_seq)
            pred_tags.append(p_seq)
        scores = tk.score_sequences(true_tags, pred_tags)
        scores.pop("scheme", None)
        return scores

    def train(self) -> None:
        from transformers import DataCollatorForTokenClassification, Trainer, TrainerCallback, TrainingArguments

        self.device_used = hft.device_name()
        MessageProtocol.status("training", "Training gestartet...")
        MessageProtocol.status("training", f"Gerät: {self.device_used.upper()}")

        total_steps = max(
            (len(self.train_dataset) // max(self.config.batch_size, 1))
            * max(self.config.epochs, 1), 1)
        if int(self.config.max_steps) > 0:
            total_steps = int(self.config.max_steps)

        args = hft.build_training_arguments(self.config, self.config.effective_output_dir(), TrainingArguments)
        self._trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=DataCollatorForTokenClassification(tokenizer=self.tokenizer),
            compute_metrics=self._compute_metrics,
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
            architecture=self.model_type, num_labels=len(self.labels),
        )
        # accuracy ist hier pro Token (inkl. O) — f1/precision/recall zaehlen
        # bei NER ganze Entitaeten. Beides ausdruecklich benennen.
        metrics["token_accuracy"] = metrics.get("accuracy", 0.0)
        metrics["metric_scheme"] = "seqeval_entity" if self._scheme == "entity" else "token_weighted"
        if self._scheme == "entity":
            metrics["entity_f1"] = metrics.get("f1", 0.0)
        MessageProtocol.status(
            "validating",
            f"F1 {metrics.get('f1', 0):.3f} | Precision {metrics.get('precision', 0):.3f} | "
            f"Recall {metrics.get('recall', 0):.3f} | Token-Accuracy {metrics.get('accuracy', 0):.3f}",
        )
        return metrics

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        (out / "label_mapping.json").write_text(json.dumps({
            "task": "token_classification",
            "classes": self.labels,
            "id2label": {str(i): l for i, l in enumerate(self.labels)},
            "scheme": self._scheme,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        tk.save_spec(out, {**self.spec, "labels": self.labels, "scheme": self._scheme,
                           "max_seq_length": int(self.config.max_seq_length or 128)})
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)
