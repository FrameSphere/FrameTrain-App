"""
Sentence Embeddings (Sentence-Transformers)
===========================================
Trainiert Embedding-Modelle fuer Suche, RAG und Aehnlichkeit. Basis ist ein
fertiges Sentence-Transformers-Modell (all-MiniLM, BGE, E5, GTE, MPNet) oder
jeder HF-Encoder — dann legt sentence-transformers Mean-Pooling darueber.

Das Datenformat bestimmt den Loss (Erkennung in ft_data.embedding):
    anchor/positive              -> MultipleNegativesRankingLoss
    anchor/positive/negative     -> MNRL mit harten Negativen
    sentence1/sentence2/score    -> CoSENTLoss (Score auf 0..1 normiert)
    text/label                   -> BatchAllTripletLoss

Gemessen wird auf dem Val-Split: bei Paaren/Tripeln Retrieval (Recall@1/@5,
MRR@10 ueber alle Val-Dokumente), bei Scores Spearman, bei Klassen die
1-NN-Accuracy. Der Wert VOR dem Training wird mitgemeldet, damit sichtbar
ist, was das Training gebracht hat.

plugin_config:
    format                        pairs | triplets | scored | labeled (sonst automatisch)
    anchor_column, positive_column, negative_column, score_column,
    text_column, label_column     eigene Spaltennamen
    query_prefix, document_prefix z.B. "query: " / "passage: " fuer E5
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft

from ft_data.splits import split_files
from ft_data import embedding as emb


def _st_import(name: str):
    """sentence-transformers 6 hat Module verschoben; alte Pfade warnen nur noch."""
    import importlib
    for mod in (f"sentence_transformers.sentence_transformer.{name}", f"sentence_transformers.{name}"):
        try:
            return importlib.import_module(mod)
        except ImportError:
            continue
    raise ImportError(f"sentence_transformers.{name} nicht gefunden")


def _dimension(model) -> int:
    """v6 heisst es get_embedding_dimension, vorher get_sentence_embedding_dimension."""
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                return int(fn() or 0)
            except Exception:
                continue
    return 0


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.eval_cols: Dict[str, List[Any]] = {}
        self.spec: Dict[str, Any] = {}
        self.model_type = ""
        self.is_st_model = False
        self.device_used = "cpu"
        self.evaluator = None
        self.baseline: Dict[str, float] = {}
        self.query_prefix = str(config.get_plugin_value("query_prefix", "") or "")
        self.document_prefix = str(config.get_plugin_value("document_prefix", "") or "")
        self._trainer = None
        self._start_time = time.time()
        self._last_train_loss = 0.0
        self._last_lr = config.learning_rate

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            MessageProtocol.error(
                "sentence-transformers fehlt",
                "Das Embedding-Plugin braucht das Paket sentence-transformers.\n\n"
                + __import__("ft_data.deps", fromlist=["install_hint"]).install_hint("sentence-transformers"),
            )
            return False
        root = Path(self.config.model_path)
        cfg_path = root / "config.json"
        if cfg_path.exists():
            try:
                self.model_type = json.loads(cfg_path.read_text(encoding="utf-8")).get("model_type", "")
            except Exception:
                self.model_type = ""
        self.is_st_model = (root / "modules.json").exists()
        MessageProtocol.status(
            "init",
            f"✓ Architektur erkannt: {self.model_type or 'unbekannt'} | "
            + ("Sentence-Transformers-Modell" if self.is_st_model
               else "HF-Encoder — Mean-Pooling wird ergaenzt"),
        )

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def _read(self, files) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for f in files:
            rows.extend(emb.load_rows(f))
        return rows

    def _prefix(self, cols: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        if not (self.query_prefix or self.document_prefix):
            return cols
        out = dict(cols)
        for key in ("anchor", "sentence1"):
            if key in out and self.query_prefix:
                out[key] = [self.query_prefix + t for t in out[key]]
        for key in ("positive", "negative", "sentence2"):
            if key in out and self.document_prefix:
                out[key] = [self.document_prefix + t for t in out[key]]
        return out

    def load_data(self) -> None:
        from datasets import Dataset

        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        files = split_files(root, emb.DATA_EXTS)
        if not files.get("train"):
            raise ValueError(f"Keine Datendateien in '{root}' gefunden (erwartet: {', '.join(emb.DATA_EXTS)}).")

        train_rows = self._read(files["train"])
        if not train_rows:
            raise ValueError("Dataset enthaelt keine Zeilen.")
        self.spec = emb.resolve_spec(list(train_rows[0].keys()), train_rows, self.config.plugin_config or {})
        MessageProtocol.status("loading_data", f"Format: {emb.describe(self.spec)}")

        if files.get("val"):
            eval_rows = self._read(files["val"])
        else:
            import random
            rng = random.Random(self.config.seed)
            rows = list(train_rows)
            rng.shuffle(rows)
            n_val = max(1, int(round(len(rows) * 0.1))) if len(rows) > 1 else 0
            eval_rows, train_rows = rows[:n_val], rows[n_val:]
            MessageProtocol.status("loading_data", "Kein Validierungs-Split gefunden — 10% abgetrennt.")

        label2id: Optional[Dict[str, int]] = {} if self.spec["format"] == "labeled" else None
        train_cols = self._prefix(emb.to_columns(train_rows, self.spec, label2id))
        eval_cols = self._prefix(emb.to_columns(eval_rows, self.spec, label2id))
        if label2id is not None:
            self.spec["labels"] = list(label2id)
        n_train = len(next(iter(train_cols.values())))
        if n_train == 0:
            raise ValueError("Keine vollstaendigen Zeilen im Training (leere Texte oder fehlende Scores?).")

        eval_ds = Dataset.from_dict(eval_cols)
        eval_ds = hft.cap_eval_dataset(eval_ds, getattr(self.config, "max_eval_samples", 0), self.config.seed)
        self.train_dataset = Dataset.from_dict(train_cols)
        self.eval_dataset = eval_ds
        self.eval_cols = {k: list(eval_ds[k]) for k in eval_ds.column_names}
        MessageProtocol.status(
            "loading_data", f"✓ Train: {len(self.train_dataset)} | Eval: {len(self.eval_dataset)}")

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.device_used = hft.device_name()
        MessageProtocol.status("building_model", "Lade Embedding-Modell...")
        self.model = SentenceTransformer(self.config.model_path, device=self.device_used)
        # Die Sequenzlaenge aus dem Formular deckeln, aber nie ueber das
        # hinaus, was das Modell kann (MiniLM: 256, BERT: 512).
        limit = int(self.config.max_seq_length or 0)
        if limit > 0 and self.model.max_seq_length:
            self.model.max_seq_length = min(limit, int(self.model.max_seq_length))
        dim = _dimension(self.model)
        params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status(
            "building_model",
            f"✓ Modell geladen | Parameter: {params/1e6:.1f}M | Dimension: {dim or '?'} | "
            f"Max. Laenge: {self.model.max_seq_length}",
        )
        self.evaluator = self._build_evaluator()
        self.baseline = self._evaluate_now("vor dem Training")

    def _build_evaluator(self):
        ev = _st_import("evaluation")
        fmt = self.spec["format"]
        bs = max(int(self.config.batch_size), 8)
        if fmt in ("pairs", "triplets"):
            queries, corpus, relevant = emb.retrieval_set(self.eval_cols)
            if not queries:
                return None
            return ev.InformationRetrievalEvaluator(
                queries=queries, corpus=corpus, relevant_docs=relevant, name="val",
                accuracy_at_k=[1, 5, 10], precision_recall_at_k=[1, 5, 10],
                mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[100],
                show_progress_bar=False, batch_size=bs, write_csv=False,
            )
        if fmt == "scored":
            c = self.eval_cols
            if len(c.get("score", [])) < 2:
                return None
            return ev.EmbeddingSimilarityEvaluator(
                c["sentence1"], c["sentence2"], c["score"], name="val",
                batch_size=bs, show_progress_bar=False, write_csv=False,
            )
        return None  # labeled: 1-NN-Accuracy rechnet _evaluate_now selbst

    def _evaluate_now(self, when: str) -> Dict[str, float]:
        """Standard-Kennzahlen jetzt — vor dem Training als Vergleichswert."""
        fmt = self.spec["format"]
        out: Dict[str, float] = {}
        if fmt == "labeled":
            texts, labels = self.eval_cols.get("text", []), self.eval_cols.get("label", [])
            if len(texts) >= 2:
                vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                out["knn_accuracy"] = emb.knn_label_accuracy(vecs, labels)
        elif self.evaluator is not None:
            raw = self.evaluator(self.model)
            out = self._standard(raw, prefix="val_")
        if out:
            MessageProtocol.status("evaluating", f"Kennzahlen {when}: " + ", ".join(
                f"{k} {v:.3f}" for k, v in out.items()))
        return out

    @staticmethod
    def _standard(raw: Dict[str, Any], prefix: str) -> Dict[str, float]:
        """Evaluator-Schluessel (val_cosine_recall@1) -> klare Namen (recall_at_1)."""
        mapping = {
            "cosine_recall@1": "recall_at_1", "cosine_recall@5": "recall_at_5",
            "cosine_recall@10": "recall_at_10", "cosine_accuracy@1": "hit_at_1",
            "cosine_mrr@10": "mrr_at_10", "cosine_ndcg@10": "ndcg_at_10",
            "spearman_cosine": "spearman", "pearson_cosine": "pearson",
        }
        out: Dict[str, float] = {}
        for key, name in mapping.items():
            for full in (f"eval_{prefix}{key}", f"{prefix}{key}", key):
                if full in raw and raw[full] is not None:
                    out[name] = float(raw[full])
                    break
        return out

    # ── 4. Training ─────────────────────────────────────────────────────────
    def _loss(self):
        losses = _st_import("losses")
        fmt = self.spec["format"]
        if fmt in ("pairs", "triplets"):
            return losses.MultipleNegativesRankingLoss(self.model)
        if fmt == "scored":
            return losses.CoSENTLoss(self.model)
        return losses.BatchAllTripletLoss(self.model)

    def train(self) -> None:
        from transformers import TrainerCallback
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        BatchSamplers = _st_import("training_args").BatchSamplers

        MessageProtocol.status("training", "Training gestartet...")
        MessageProtocol.status("training", f"Gerät: {self.device_used.upper()}")
        fmt = self.spec["format"]
        # MNRL nutzt die anderen Paare im Batch als Negative. Steht dieselbe
        # Antwort zweimal im Batch, wuerde sie als "falsch" bestraft.
        # BatchAllTripletLoss braucht je Batch mehrere Texte derselben Klasse.
        sampler = {"pairs": BatchSamplers.NO_DUPLICATES, "triplets": BatchSamplers.NO_DUPLICATES,
                   "labeled": BatchSamplers.GROUP_BY_LABEL}.get(fmt, BatchSamplers.BATCH_SAMPLER)

        total_steps = max(
            (len(self.train_dataset) // max(self.config.batch_size, 1))
            * max(self.config.epochs, 1), 1)
        if int(self.config.max_steps) > 0:
            total_steps = int(self.config.max_steps)

        args = hft.build_training_arguments(
            self.config, self.config.effective_output_dir(), SentenceTransformerTrainingArguments,
            batch_sampler=sampler,
        )
        self._trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset if len(self.eval_dataset) else None,
            loss=self._loss(),
            evaluator=self.evaluator,
            callbacks=[hft.progress_callback(TrainerCallback, self, total_steps)],
        )
        self._start_time = time.time()
        self._trainer.train()
        MessageProtocol.status("training", "Training abgeschlossen")

    # ── 5. Validierung ──────────────────────────────────────────────────────
    def validate(self) -> Dict[str, float]:
        MessageProtocol.status("validating", "Finale Validierung...")
        result = self._trainer.evaluate() if self.eval_dataset is not None and len(self.eval_dataset) else {}
        metrics = hft.final_metrics(
            self, self._trainer, result, self._start_time,
            architecture=self.model_type, num_labels=len(self.spec.get("labels") or []),
        )
        # Klassifikations-Kennzahlen haben hier keine Bedeutung — Nullen
        # wuerden auf der Analyse-Seite wie ein kaputtes Modell aussehen.
        for key in ("accuracy", "f1", "precision", "recall"):
            metrics.pop(key, None)
        if not self.spec.get("labels"):
            metrics.pop("num_labels", None)

        now = self._standard(result, prefix="val_")
        if self.spec["format"] == "labeled":
            now = self._evaluate_now("nach dem Training")
        metrics.update(now)
        for k, v in self.baseline.items():
            metrics[f"{k}_before"] = v
        # accuracy = der Wert, der fuer das Format am meisten sagt: Treffer auf
        # Platz 1 bei Retrieval, 1-NN bei Klassen. Bei Scores gibt es keinen.
        if "hit_at_1" in now:
            metrics["accuracy"] = now["hit_at_1"]
        elif "knn_accuracy" in now:
            metrics["accuracy"] = now["knn_accuracy"]
        metrics["embedding_format"] = self.spec["format"]
        metrics["embedding_dim"] = _dimension(self.model)
        if now:
            before = ", ".join(f"{k} {self.baseline[k]:.3f} -> {now[k]:.3f}" for k in now if k in self.baseline)
            MessageProtocol.status("validating", f"Vorher -> nachher: {before or ', '.join(f'{k} {v:.3f}' for k, v in now.items())}")
        return metrics

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        # ST-Format (modules.json, Pooling) — so laden Test, Modell-Server und
        # jede andere Anwendung das Modell mit SentenceTransformer(pfad).
        self.model.save(str(out))
        emb.save_spec(out, {**self.spec, "query_prefix": self.query_prefix,
                            "document_prefix": self.document_prefix})
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)
