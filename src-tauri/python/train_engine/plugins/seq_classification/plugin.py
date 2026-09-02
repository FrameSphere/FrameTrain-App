"""
plugins/seq_classification/plugin.py
=====================================
XLM-RoBERTa / BERT / DeBERTa Sequenzklassifikations-Plugin.

Unterstützte Architekturen:
  xlm-roberta, roberta, bert, deberta, deberta-v2, distilbert,
  albert, camembert, electra, rembert, xlm

Nicht unterstützte Architekturen führen zu einem klaren Fehlermessage.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

# ─── Import Guard ─────────────────────────────────────────────────────────────
try:
    import torch
    import numpy as np
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoConfig,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    from datasets import load_dataset, Dataset as HFDataset
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        f1_score,
    )
except ImportError as e:
    # Wird durch train_engine.py abgefangen
    raise

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.plugin_base import TrainPlugin
from core.config import TrainingConfig
from core.protocol import MessageProtocol

# ─── Unterstützte Architekturen ───────────────────────────────────────────────
SUPPORTED_ARCHITECTURES = {
    "xlm-roberta", "roberta", "bert", "deberta", "deberta-v2",
    "distilbert", "albert", "camembert", "electra", "rembert",
    "xlm", "ernie", "funnel", "mpnet", "squeezebert", "layoutlm",
}

# ─── Dataset-Formate ──────────────────────────────────────────────────────────
LABEL_COLUMN_NAMES = ["label", "labels", "category", "class", "target", "sentiment"]
TEXT_COLUMN_NAMES  = ["text", "sentence", "content", "review_body", "input", "document", "title", "body", "description", "abstract", "question", "passage", "premise", "hypothesis"]
# Spalten die nie als Text-Eingabe verwendet werden sollen
ID_COLUMN_NAMES = {
    "id", "idx", "index", "row_id", "sample_id", "uid", "uuid", "key",
    "review_id", "product_id", "user_id", "item_id", "doc_id", "article_id",
    "tweet_id", "post_id", "comment_id", "message_id", "conversation_id",
    "passage_id", "question_id", "answer_id", "sentence_id", "token_id",
    "source_id", "target_id", "pair_id", "example_id", "data_id",
}


def _detect_columns(features: dict):
    """Erkennt automatisch Text- und Label-Spalte."""
    # Interne Spalten wie '__index_level_0__' (Pandas/Parquet-Artefakt) sind
    # NIE Text oder Label — vorher wurde genau diese Index-Spalte als "Label"
    # erkannt (KeyError beim Eval-Split, tausende Pseudo-Klassen).
    cols = [c for c in features.keys() if not c.startswith("__")]
    label_col = None
    text_col  = None

    for name in LABEL_COLUMN_NAMES:
        if name in cols:
            label_col = name
            break

    for name in TEXT_COLUMN_NAMES:
        if name in cols:
            text_col = name
            break

    # Fallback: erste nicht-ID Spalte ist Text, letzte ist Label
    if text_col is None or label_col is None:
        non_id_cols = [c for c in cols if c.lower() not in ID_COLUMN_NAMES]
        if text_col is None and non_id_cols:
            text_col = non_id_cols[0]
        if label_col is None and len(non_id_cols) >= 2:
            label_col = non_id_cols[-1]
        elif label_col is None and len(cols) >= 2:
            label_col = cols[-1]

    return text_col, label_col


class Plugin(TrainPlugin):
    """Sequenzklassifikations-Plugin für XLM-RoBERTa & ähnliche Encoder."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.tokenizer     = None
        self.model         = None
        self.train_dataset = None
        self.eval_dataset  = None
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.text_col:  str = "text"
        self.label_col: str = "label"
        self.num_labels: int = 2
        self._start_time      = time.time()
        self._last_train_loss: float = 0.0
        self._last_lr:         float = 0.0
        self._last_step:       int   = 0

    # ─── 1. Setup ──────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """Prüft Modell-Architektur und initialisiert Tokenizer."""
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modell-Verzeichnis nicht gefunden: {model_path}\n"
                "Bitte importiere zuerst ein Modell in FrameTrain."
            )

        cfg_file = model_path / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(
                f"Keine config.json im Modell-Verzeichnis: {model_path}\n"
                "Das Verzeichnis enthält kein gültiges HuggingFace-Modell."
            )

        with open(cfg_file, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)

        model_type = model_cfg.get("model_type", "").lower()
        # Fuer die Analyse-Seite: echte Architektur statt hartkodiertem Platzhalter.
        self.model_type = model_type

        if model_type not in SUPPORTED_ARCHITECTURES:
            supported_list = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
            raise ValueError(
                f"✗ Modell-Architektur '{model_type}' wird noch nicht unterstützt.\n\n"
                f"Unterstützte Architekturen:\n  {supported_list}\n\n"
                f"Bitte verwende ein XLM-RoBERTa, BERT oder ähnliches Encoder-Modell."
            )

        MessageProtocol.status(
            "init",
            f"✓ Architektur erkannt: {model_type} | Lade Tokenizer..."
        )

        # Tokenizer laden
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )

        MessageProtocol.status("init", "Tokenizer geladen ✓")
        # WICHTIG: Der Orchestrator wertet den Rückgabewert aus — ohne
        # "return True" (implizit None) brach das Training hier STILL ab.
        return True

    # ─── 2. Daten laden ────────────────────────────────────────────────────

    def load_data(self) -> None:
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset nicht gefunden: {dataset_path}")

        MessageProtocol.status("loading_data", f"Lade Dataset: {dataset_path.name}")

        # Erkennung: einzelne Datei vs. Verzeichnis
        if dataset_path.is_dir():
            raw = self._load_from_dir(dataset_path)
        else:
            raw = self._load_from_file(dataset_path)

        # Spalten erkennen
        split = list(raw.keys())[0]
        self.text_col, self.label_col = _detect_columns(raw[split].features)

        if self.text_col is None or self.label_col is None:
            raise ValueError(
                f"Konnte Text- und Label-Spalten nicht erkennen.\n"
                f"Gefundene Spalten: {list(raw[split].features.keys())}\n"
                f"Erwartet z.B.: 'text' / 'label'"
            )

        MessageProtocol.status(
            "loading_data",
            f"Text-Spalte: '{self.text_col}', Label-Spalte: '{self.label_col}'"
        )

        # Labels normalisieren — aus ALLEN Splits sammeln, nicht nur dem ersten.
        # Vorher: Labels nur aus train → unbekanntes Label im Eval-Split = KeyError.
        unique_labels = set()
        total_rows = 0
        for split_name in raw.keys():
            if self.label_col not in raw[split_name].features:
                continue
            total_rows += len(raw[split_name])
            for val in raw[split_name][self.label_col]:
                if isinstance(val, list):
                    # Wenn die Label eine Liste sind, take the first element
                    if val:
                        unique_labels.add(val[0])
                else:
                    unique_labels.add(val)
        all_labels = sorted(unique_labels, key=lambda x: str(x))
        self.label2id = {str(l): i for i, l in enumerate(all_labels)}
        self.id2label = {i: str(l) for i, l in enumerate(all_labels)}
        self.num_labels = len(all_labels)

        # Sanity-Check: weniger als 2 Klassen = keine Klassifikation. Ohne diesen
        # Check setzt HuggingFace bei num_labels=1 automatisch auf Regression um
        # (MSELoss) und stirbt dann an Integer-Labels mit einem völlig
        # irreführenden "mse_loss_out_mps: only defined for floating types".
        if self.num_labels < 2:
            only = all_labels[0] if all_labels else "—"
            raise ValueError(
                f"Label-Spalte '{self.label_col}' enthält nur einen einzigen Wert "
                f"({only!r}) bei {total_rows} Zeilen — damit lässt sich keine "
                "Klassifikation trainieren.\n\n"
                "Häufigste Ursache: Es wurde ein Split ohne Labels importiert "
                "(z.B. der 'unsupervised'-Split von IMDB, dort ist label immer -1).\n\n"
                "Lösungen:\n"
                "  - Dataset neu importieren und den train/test-Split wählen\n"
                "  - Prüfen, ob die richtige Spalte als Label erkannt wurde "
                f"(erkannt: '{self.label_col}', vorhanden: {list(raw[split].features.keys())})"
            )

        # Sanity-Check: fast so viele "Klassen" wie Zeilen = ID-/Index-Spalte,
        # kein Klassifikations-Label. Klare Meldung statt kryptischem Crash.
        if self.num_labels > 1000 or (total_rows > 0 and self.num_labels > total_rows * 0.5):
            raise ValueError(
                f"Label-Spalte '{self.label_col}' hat {self.num_labels} verschiedene Werte "
                f"bei {total_rows} Zeilen — das sieht nach einer ID-/Index-Spalte aus, "
                "nicht nach Klassifikations-Labels.\n"
                f"Gefundene Spalten: {list(raw[split].features.keys())}\n\n"
                "Lösungen:\n"
                "  - Dataset mit echter Label-Spalte verwenden (z.B. 'label' mit wenigen Klassen)\n"
                "  - Dieses Dataset ist evtl. kein Klassifikations-Dataset "
                "(z.B. Text-Generierung/Keyphrases) — dann passt der Task-Typ nicht"
            )

        MessageProtocol.status("loading_data", f"Labels ({self.num_labels}): {all_labels[:10]}")

        # Train/Eval-Split
        if "train" in raw and "test" in raw:
            train_raw = raw["train"]
            eval_raw  = raw["test"]
        elif "train" in raw and "validation" in raw:
            train_raw = raw["train"]
            eval_raw  = raw["validation"]
        elif "train" in raw:
            split_ds = raw["train"].train_test_split(test_size=0.1, seed=self.config.seed)
            train_raw = split_ds["train"]
            eval_raw  = split_ds["test"]
        else:
            split_ds = raw[split].train_test_split(test_size=0.1, seed=self.config.seed)
            train_raw = split_ds["train"]
            eval_raw  = split_ds["test"]

        MessageProtocol.status(
            "loading_data",
            f"Train: {len(train_raw)} | Eval: {len(eval_raw)} | Tokenisiere..."
        )

        def tokenize_fn(batch):
            tokens = self.tokenizer(
                batch[self.text_col],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_seq_length,
            )
            # Handle list values in label column
            labels = []
            for l in batch[self.label_col]:
                if isinstance(l, list):
                    label_val = str(l[0]) if l else "0"
                else:
                    label_val = str(l)
                labels.append(self.label2id[label_val])
            tokens["labels"] = labels
            return tokens

        keep_cols = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in self.tokenizer.model_input_names:
            keep_cols.append("token_type_ids")

        self.train_dataset = train_raw.map(
            tokenize_fn, batched=True,
            remove_columns=train_raw.column_names
        )
        # Eval-Split optional deckeln. Ohne das kostet ein kurzer Testlauf
        # (Max Steps = 40) eine Auswertung ueber den kompletten Split — bei
        # imdb sind das 25.000 Beispiele pro Evaluation.
        max_eval = int(getattr(self.config, "max_eval_samples", 0) or 0)
        if 0 < max_eval < len(eval_raw):
            MessageProtocol.status(
                "loading_data",
                f"Eval-Split auf {max_eval} von {len(eval_raw)} Beispielen begrenzt "
                "(Max Eval Samples).",
            )
            # Zufaellig ziehen, nicht die ersten N: viele Splits sind nach
            # Label sortiert (imdb: erst 12.500x Klasse 0, dann Klasse 1).
            # Ein Praefix haette nur eine Klasse enthalten — die Auswertung
            # meldete dann Accuracy 0.0 statt eines brauchbaren Werts.
            eval_raw = eval_raw.shuffle(seed=self.config.seed).select(range(max_eval))

        self.eval_dataset = eval_raw.map(
            tokenize_fn, batched=True,
            remove_columns=eval_raw.column_names
        )
        self.train_dataset.set_format("torch")
        self.eval_dataset.set_format("torch")

        MessageProtocol.status("loading_data", "✓ Dataset tokenisiert")

    def _load_from_file(self, path: Path) -> dict:
        ext = path.suffix.lower()
        if ext == ".json":
            return load_dataset("json", data_files=str(path))
        if ext in (".jsonl", ".ndjson"):
            return load_dataset("json", data_files=str(path))
        if ext == ".csv":
            return load_dataset("csv", data_files=str(path))
        if ext in (".parquet", ".pq"):
            return load_dataset("parquet", data_files=str(path))
        raise ValueError(f"Nicht unterstütztes Dateiformat: {ext}")

    def _load_from_dir(self, path: Path) -> dict:
        # Strategie 1: Suche nach train/val/test-Unterordnern
        train_subdir = None
        val_subdir   = None
        test_subdir  = None

        for subdir in path.iterdir():
            if subdir.is_dir():
                n = subdir.name.lower()
                if   n in ("train", "training"):                  train_subdir = subdir
                elif n in ("val", "validation", "valid", "dev"): val_subdir   = subdir
                elif n in ("test", "testing", "eval"):           test_subdir  = subdir

        SKIP = {"dataset_infos.json", "metadata.json"}

        def _files(subdir, glob):
            """Alle Dateien eines Globs in einem Unterordner, ohne Metadateien."""
            if subdir is None:
                return []
            return [f for f in sorted(subdir.glob(glob)) if f.name not in SKIP]

        if train_subdir:
            for ext_glob, ext_key in [
                ("*.parquet", "parquet"),
                ("*.jsonl",   "json"),
                ("*.json",    "json"),
                ("*.csv",     "csv"),
            ]:
                train_files = _files(train_subdir, ext_glob)
                if not train_files:
                    continue

                val_files  = _files(val_subdir,  ext_glob)
                test_files = _files(test_subdir, ext_glob)

                # WICHTIG: alle Shards uebergeben, nicht nur [0]!
                # load_dataset akzeptiert eine Liste von Dateipfaden pro Split
                # und konkateniert sie automatisch (Multi-Shard-Support).
                data_files: dict = {"train": [str(f) for f in train_files]}
                if val_files:  data_files["validation"] = [str(f) for f in val_files]
                if test_files: data_files["test"]       = [str(f) for f in test_files]

                MessageProtocol.status(
                    "loading_data",
                    f"Split-Ordner gefunden | {ext_key} | "
                    f"train: {len(train_files)} Datei(en), "
                    f"val: {len(val_files)}, test: {len(test_files)}"
                )
                return load_dataset(ext_key, data_files=data_files)

        # Strategie 2: Dateien mit train/test im Namen im Root (Fallback)
        for ext_glob, ext_key in [
            ("*.parquet", "parquet"),
            ("*.jsonl",   "json"),
            ("*.json",    "json"),
            ("*.csv",     "csv"),
        ]:
            files = [f for f in sorted(path.glob(ext_glob)) if f.name not in SKIP]
            if not files:
                continue

            train_files = [f for f in files if "train" in f.stem.lower()]
            test_files  = [f for f in files if any(k in f.stem.lower() for k in ("test", "eval", "val"))]

            if train_files and test_files:
                MessageProtocol.status("loading_data",
                    f"Root-Dateien | {ext_key} | train: {len(train_files)}, test: {len(test_files)}")
                return load_dataset(ext_key, data_files={
                    "train": [str(f) for f in train_files],
                    "test":  [str(f) for f in test_files],
                })

            # Alle Dateien als ungeteiltes Dataset
            if files:
                MessageProtocol.status("loading_data",
                    f"Root-Dateien | {ext_key} | {len(files)} Datei(en) gesamt")
                return load_dataset(ext_key,
                    data_files={"train": [str(f) for f in files]})

        raise FileNotFoundError(
            f"Keine unterstuetzten Dataset-Dateien in {path} gefunden.\n"
            f"Erwartet: .parquet, .jsonl, .json oder .csv in train/val/test-Ordnern "
            f"oder direkt im Root."
        )

    # ─── 3. Modell laden ───────────────────────────────────────────────────

    def build_model(self) -> None:
        model_path = Path(self.config.model_path)

        MessageProtocol.status("building_model", "Lade Modell für Sequenzklassifikation...")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )

        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        MessageProtocol.status(
            "building_model",
            f"✓ Modell geladen | Parameter: {param_count:.1f}M | Labels: {self.num_labels}"
        )

    # ─── 4. Training ───────────────────────────────────────────────────────

    def train(self) -> None:
        output_dir = Path(self.config.effective_output_dir())
        output_dir.mkdir(parents=True, exist_ok=True)

        # Gerät bestimmen
        if torch.cuda.is_available():
            device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info = "MPS (Apple Silicon)"
        else:
            device_info = "CPU"

        # Merken, damit die Analyse-Seite das echte Geraet zeigen kann statt es
        # aus den fp16/bf16-Flags zu raten.
        self.device_used = (
            "cuda" if torch.cuda.is_available()
            else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            else "cpu"
        )

        MessageProtocol.status("training", f"Gerät: {device_info}")

        total_steps = (
            (len(self.train_dataset) // self.config.batch_size)
            * self.config.epochs
            // max(self.config.gradient_accumulation_steps, 1)
        )
        # Max Steps deckelt die Gesamtzahl — sonst zeigt die UI "Step 20 / 9375",
        # obwohl das Training nach 60 Schritten endet.
        if int(self.config.max_steps) > 0:
            total_steps = int(self.config.max_steps)

        # ── Progress-Callback ──────────────────────────────────────────────
        plugin_ref = self

        from transformers import TrainerCallback

        class ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                if plugin_ref.is_stopped:
                    control.should_training_stop = True
                    return
                # Der Trainer kennt die tatsaechliche Schrittzahl am genauesten
                # (beruecksichtigt drop_last, max_steps, verteiltes Training).
                nonlocal_total = getattr(state, "max_steps", 0)

                # Eval-Logs und Train-Logs separat behandeln
                is_eval = "eval_loss" in logs
                if is_eval:
                    # Nur val_loss + eval metrics senden, train_loss unverändert lassen
                    v_loss = logs.get("eval_loss")
                    extra_metrics = {
                        k: v for k, v in logs.items()
                        if k not in ("eval_loss", "epoch", "eval_runtime",
                                     "eval_samples_per_second", "eval_steps_per_second")
                        and isinstance(v, (int, float))
                    }
                    MessageProtocol.progress(
                        epoch=max(1, min(int(plugin_ref.config.epochs), int(float(state.epoch or 0)) + 1)),
                        total_epochs=plugin_ref.config.epochs,
                        step=state.global_step,
                        total_steps=nonlocal_total or total_steps,
                        train_loss=plugin_ref._last_train_loss,
                        val_loss=v_loss,
                        learning_rate=plugin_ref._last_lr,
                        metrics=extra_metrics,
                    )
                else:
                    step   = state.global_step
                    # 1-basiert und gedeckelt: sonst erzeugte das letzte Log bei
                    # epoch=3.0 eine vierte Epoche in der Analyse.
                    epoch  = max(1, min(int(plugin_ref.config.epochs), int(float(state.epoch or 0)) + 1))
                    t_loss = logs.get("loss", logs.get("train_loss", 0.0))
                    lr     = logs.get("learning_rate", plugin_ref.config.learning_rate)
                    plugin_ref._last_train_loss = t_loss
                    plugin_ref._last_lr = lr
                    extra_metrics = {
                        k: v for k, v in logs.items()
                        if k not in ("loss", "train_loss", "learning_rate",
                                     "epoch", "step", "total_flos", "train_runtime",
                                     "train_samples_per_second", "train_steps_per_second")
                        and isinstance(v, (int, float))
                    }
                    MessageProtocol.progress(
                        epoch=epoch,
                        total_epochs=plugin_ref.config.epochs,
                        step=step,
                        total_steps=nonlocal_total or total_steps,
                        train_loss=t_loss,
                        val_loss=None,
                        learning_rate=lr,
                        metrics=extra_metrics,
                    )

            def on_step_end(self, args, state, control, **kwargs):
                # Eine Evaluation ueber einen grossen Split dauert laenger als
                # das ganze Training mit Max Steps. Ohne diese Meldung stand
                # der Fortschritt minutenlang still und sah aus wie ein Absturz.
                if control.should_evaluate and plugin_ref.eval_dataset is not None:
                    MessageProtocol.status(
                        "evaluating",
                        f"Evaluierung laeuft ueber {len(plugin_ref.eval_dataset)} "
                        f"Beispiele - das kann bei grossen Datasets dauern.",
                    )

            def on_evaluate(self, args, state, control, **kwargs):
                if not plugin_ref.is_stopped:
                    MessageProtocol.status("training", "Evaluierung abgeschlossen, Training laeuft weiter.")

            def on_epoch_end(self, args, state, control, **kwargs):
                if plugin_ref.is_stopped:
                    control.should_training_stop = True

        # ── TrainingArguments ──────────────────────────────────────────────
        use_fp16 = self.config.fp16 and torch.cuda.is_available()
        use_bf16 = (
            self.config.bf16
            and (torch.cuda.is_available() or
                 (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
        )

        # warmup: ratio hat Vorrang, steps nur wenn ratio=0
        warmup_steps = self.config.warmup_steps if self.config.warmup_ratio == 0 else 0
        warmup_ratio = self.config.warmup_ratio if warmup_steps == 0 else 0.0

        # Optimizer-Name der UI auf HuggingFace-Bezeichner abbilden.
        # Ohne diese Zuordnung lief jedes Training stillschweigend mit AdamW,
        # egal was in der UI ausgewaehlt war.
        optim_name = {
            "adamw": "adamw_torch",
            "adam": "adamw_torch",   # HF kennt kein reines Adam; AdamW ist der Ersatz
            "sgd": "sgd",
            "adafactor": "adafactor",
        }.get(str(self.config.optimizer).lower(), "adamw_torch")

        # max_steps: HuggingFace ueberschreibt damit num_train_epochs.
        # -1 (Default) = alle Epochen durchlaufen.
        max_steps = int(self.config.max_steps) if int(self.config.max_steps) > 0 else -1

        # eval_steps nur sinnvoll, wenn schrittbasiert evaluiert wird
        eval_steps = (
            max(int(self.config.eval_steps), 1)
            if str(self.config.eval_strategy).lower() == "steps"
            else None
        )

        # TrainingArguments-Felder aendern sich zwischen transformers-Versionen
        # (5.x hat z.B. group_by_length entfernt). Unbekannte Argumente werden
        # daher verworfen statt das Training mit einem TypeError abzubrechen.
        ta_kwargs = dict(
            output_dir=str(output_dir),
            num_train_epochs=self.config.epochs,
            max_steps=max_steps,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            optim=optim_name,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
            lr_scheduler_type=self.config.scheduler,
            max_grad_norm=self.config.max_grad_norm,
            label_smoothing_factor=self.config.label_smoothing,
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            eval_strategy=self.config.eval_strategy,
            **({"eval_steps": eval_steps} if eval_steps is not None else {}),
            # group_by_length nur senden, wenn aktiv. transformers 5.x kennt das
            # Feld nicht mehr; der Default (False) wuerde sonst bei jedem Lauf die
            # "...werden ignoriert: group_by_length"-Warnung ausloesen, obwohl der
            # Nutzer nichts eingestellt hat. Nur wenn bewusst aktiviert, ist die
            # Warnung (auf 5.x) berechtigt.
            **({"group_by_length": True} if self.config.group_by_length else {}),
            # Checkpointing: immer steps-basiert damit stop_training() einen
            # Checkpoint findet, auch wenn keine Epoche abgeschlossen wurde.
            # save_total_limit=2: aktuellsten + einen Backup halten,
            # aeltere werden automatisch geloescht.
            save_strategy="steps",
            save_steps=max(self.config.save_steps, 50),
            save_total_limit=2,
            logging_steps=self.config.logging_steps,
            seed=self.config.seed,
            # Dataloader
            dataloader_num_workers=0,   # MPS-sicher
            dataloader_pin_memory=False, # MPS-sicher
            dataloader_drop_last=self.config.dataloader_drop_last,
            # Logging
            report_to=[],
            disable_tqdm=True,
            load_best_model_at_end=False,  # False weil save_strategy != eval_strategy
        )

        import inspect as _inspect
        _supported = set(_inspect.signature(TrainingArguments.__init__).parameters)
        _dropped = sorted(k for k in ta_kwargs if k not in _supported)
        if _dropped:
            MessageProtocol.status(
                "training",
                "Hinweis: diese Einstellungen kennt die installierte "
                f"transformers-Version ({__import__('transformers').__version__}) "
                f"nicht und werden ignoriert: {', '.join(_dropped)}",
            )
            for k in _dropped:
                ta_kwargs.pop(k, None)

        training_args = TrainingArguments(**ta_kwargs)

        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8 if (use_fp16 or use_bf16) else None,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, preds)
            p, r, f1, _ = precision_recall_fscore_support(
                labels, preds, average="weighted", zero_division=0
            )
            return {"accuracy": acc, "f1": f1, "precision": p, "recall": r}

        self._trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[ProgressCallback()],
        )

        self._train_result = self._trainer.train()
        self._start_time_saved = time.time()

    # ─── 5. Validierung ────────────────────────────────────────────────────

    def validate(self) -> Dict[str, float]:
        MessageProtocol.status("validating", "Finale Evaluation...")
        eval_result = self._trainer.evaluate()

        # Trainings-Metriken aus Trainer
        tr = self._train_result
        total_steps  = tr.global_step if hasattr(tr, "global_step") else 0
        train_loss   = tr.training_loss if hasattr(tr, "training_loss") else 0.0

        # Beste Epoche
        best_epoch = getattr(self._trainer.state, "best_epoch", 0) or 0
        if best_epoch == 0 and hasattr(self._trainer.state, "log_history"):
            # Aus log_history extrahieren
            eval_losses = [
                (e.get("epoch", 0), e.get("eval_loss", float("inf")))
                for e in self._trainer.state.log_history
                if "eval_loss" in e
            ]
            if eval_losses:
                best_epoch = int(min(eval_losses, key=lambda x: x[1])[0])

        duration = int(time.time() - self._start_time)

        # Tatsaechlich durchlaufene Epochen. Bei Max Steps bricht das Training
        # mitten in einer Epoche ab — die Analyse-Seite meldete dann trotzdem
        # die geplanten 3 Epochen.
        epochs_done = getattr(self._trainer.state, "epoch", None)
        total_epochs = (
            max(1, round(float(epochs_done)))
            if epochs_done is not None
            else self.config.epochs
        )

        return {
            "final_train_loss": float(train_loss),
            "final_val_loss":   float(eval_result.get("eval_loss", 0.0)),
            "accuracy":         float(eval_result.get("eval_accuracy", 0.0)),
            "f1":               float(eval_result.get("eval_f1", 0.0)),
            "precision":        float(eval_result.get("eval_precision", 0.0)),
            "recall":           float(eval_result.get("eval_recall", 0.0)),
            "total_epochs":     int(total_epochs),
            "total_steps":      total_steps,
            "best_epoch":       best_epoch,
            "training_duration_seconds": duration,
            # Ohne diese Felder zeigte die Analyse-Seite fest "xlm-roberta"
            # und n_train/n_val = 0 an, egal was tatsaechlich trainiert wurde.
            "architecture":     getattr(self, "model_type", "") or "unbekannt",
            "num_labels":       int(self.num_labels),
            "n_train":          int(len(self.train_dataset)) if self.train_dataset is not None else 0,
            "n_val":            int(len(self.eval_dataset)) if self.eval_dataset is not None else 0,
            "device":           getattr(self, "device_used", "cpu"),
        }

    # ─── 6. Export ─────────────────────────────────────────────────────────

    def export(self) -> str:
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._trainer.save_model(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

        # Label-Mapping sichern
        label_map = {"label2id": self.label2id, "id2label": self.id2label}
        with open(output_path / "label_mapping.json", "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)

        MessageProtocol.status("saving", f"✓ Modell gespeichert: {output_path}")
        return str(output_path)
