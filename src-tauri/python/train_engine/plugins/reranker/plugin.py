"""
Reranker / Cross-Encoder (Sentence-Transformers)
================================================
Ein Cross-Encoder liest Frage und Passage zusammen und gibt eine Relevanz aus.
Er ist langsamer als ein Embedding-Modell, ordnet die Top-Treffer einer Suche
aber deutlich genauer — typischer zweiter Schritt in RAG-Pipelines.

Basis: ein fertiger Reranker (cross-encoder/ms-marco-MiniLM-L-6-v2,
BAAI/bge-reranker-base) oder jeder HF-Encoder; der bekommt einen frischen
Kopf mit einer Ausgabe.

Daten (ft_data.rerank): Paare mit Score/Label oder Frage + positive [+ negative]
Passagen. Alles wird zu (Frage, Passage, Relevanz 0..1) und mit
BinaryCrossEntropyLoss trainiert. Bei Tripeln ohne genug Negative werden
Passagen anderer Fragen als Negative gezogen (plugin_config.negatives_per_query).

Gemessen wird je Val-Frage ueber ihre Kandidaten: MRR@10, nDCG@10, Hit@1;
bei Score-Paaren zusaetzlich Accuracy und Spearman — vor und nach dem Training.

plugin_config:
    query_column, passage_column, score_column, positive_column, negative_column
    negatives_per_query   Standard 3 (Training); die Bewertung nutzt bis zu 9
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft

from ft_data.splits import split_files
from ft_data import rerank as R

SUPPORTED_ARCHITECTURES = {
    "bert", "distilbert", "roberta", "xlm-roberta", "deberta", "deberta-v2",
    "electra", "albert", "camembert", "mpnet", "modernbert",
}
EVAL_NEGATIVES = 9


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.pc = config.plugin_config or {}
        self.model = None
        self.model_type = ""
        self.spec: Dict[str, Any] = {}
        self.train_groups: Dict[str, Dict[str, float]] = {}
        self.eval_groups: Dict[str, Dict[str, float]] = {}
        self.train_dataset = None
        self.eval_dataset = None
        self.baseline: Dict[str, float] = {}
        self.device_used = "cpu"
        self._trainer = None
        self._start_time = time.time()
        self._last_train_loss = 0.0
        self._last_lr = config.learning_rate

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder  # noqa: F401
        except ImportError:
            from ft_data.deps import install_hint
            MessageProtocol.error("sentence-transformers fehlt",
                                  "Der Reranker braucht das Paket sentence-transformers (ab Version 4).\n\n"
                                  + install_hint("sentence-transformers"))
            return False
        cfg = Path(self.config.model_path) / "config.json"
        if cfg.exists():
            try:
                self.model_type = str(json.loads(cfg.read_text(encoding="utf-8")).get("model_type", ""))
            except Exception:
                pass
        if self.model_type and self.model_type not in SUPPORTED_ARCHITECTURES:
            MessageProtocol.warning(f"Architektur '{self.model_type}' ist fuer Reranker nicht geprueft — es wird trotzdem versucht.")
        fresh = not R.single_head(Path(self.config.model_path))
        MessageProtocol.status(
            "init", f"Architektur: {self.model_type or 'unbekannt'} | "
                    + ("Encoder ohne Reranker-Kopf — ein Kopf mit einer Ausgabe wird neu angelegt"
                       if fresh else "fertiger Reranker — wird weitertrainiert"))

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def load_data(self) -> None:
        from datasets import Dataset

        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        files = split_files(root, R.DATA_EXTS)
        if not files.get("train"):
            raise ValueError(f"Keine Datendateien in '{root}' gefunden (erwartet: {', '.join(R.DATA_EXTS)}).")
        rows = R.load_files(files["train"])
        if not rows:
            raise ValueError("Dataset enthaelt keine Zeilen.")
        self.spec = R.resolve_spec(list(rows[0].keys()), rows, self.pc)
        MessageProtocol.status("loading_data", f"Format: {R.describe(self.spec)}")
        grp = R.groups(rows, self.spec)
        if files.get("val"):
            val = R.groups(R.load_files(files["val"]), self.spec)
        else:
            grp, val = R.split_groups(grp, 0.1, self.config.seed)
            if val:
                MessageProtocol.status("loading_data", f"Kein Validierungs-Split gefunden — {len(val)} Fragen abgetrennt.")
            else:
                val = dict(grp)
                MessageProtocol.status("loading_data", "Sehr kleines Dataset — Validierung auf den Trainingsdaten (nur Lernnachweis).")
        cap = int(getattr(self.config, "max_eval_samples", 0) or 0)
        if cap and len(val) > cap:
            keep = sorted(val)[:cap]
            val = {q: val[q] for q in keep}
            MessageProtocol.status("loading_data", f"Val-Split auf {cap} Fragen begrenzt (Max Eval Samples).")
        if self.spec["format"] == "triplets":
            n_neg = int(self.pc.get("negatives_per_query", 3) or 3)
            grp = R.add_sampled_negatives(grp, n_neg, self.config.seed)
            val = R.add_sampled_negatives(val, EVAL_NEGATIVES, self.config.seed + 1)
        self.train_groups, self.eval_groups = grp, val
        self.train_dataset = Dataset.from_dict(R.to_pairs(grp))
        self.eval_dataset = Dataset.from_dict(R.to_pairs(val))
        if not len(self.train_dataset):
            raise ValueError("Keine vollstaendigen Paare im Training.")
        pos = sum(1 for l in self.train_dataset["label"] if l >= 0.5)
        MessageProtocol.status(
            "loading_data",
            f"✓ Train: {len(grp)} Fragen, {len(self.train_dataset)} Paare ({pos} relevant) | "
            f"Val: {len(val)} Fragen, {len(self.eval_dataset)} Paare")

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        from sentence_transformers.cross_encoder import CrossEncoder

        self.device_used = hft.device_name()
        MessageProtocol.status("building_model", "Lade Cross-Encoder...")
        kwargs: Dict[str, Any] = {"device": self.device_used}
        if not R.single_head(Path(self.config.model_path)):
            kwargs["num_labels"] = 1
        limit = int(self.config.max_seq_length or 0)
        if limit > 0:
            kwargs["max_length"] = max(limit, 64)
        self.model = CrossEncoder(str(self.config.model_path), **kwargs)
        params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status("building_model", f"✓ Modell geladen | {params/1e6:.1f}M Parameter | Gerät: {self.device_used.upper()}")
        self.baseline = self.evaluate_now("vor dem Training")

    def score(self, pairs: List[Any]) -> List[float]:
        if not pairs:
            return []
        out = self.model.predict(pairs, batch_size=max(int(self.config.batch_size), 16),
                                 show_progress_bar=False, convert_to_numpy=True)
        return [float(x) for x in out]

    def evaluate_now(self, when: str) -> Dict[str, float]:
        m, _ = R.evaluate(self.score, self.eval_groups, self.spec.get("format", "scored"))
        m.pop("ranked_queries", None)
        if m:
            MessageProtocol.status("evaluating", f"Kennzahlen {when}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items()))
        return m

    # ── 4. Training ─────────────────────────────────────────────────────────
    def train(self) -> None:
        from transformers import TrainerCallback
        from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

        total = max(1, len(self.train_dataset) // max(int(self.config.batch_size), 1)) * max(1, int(self.config.epochs))
        if int(self.config.max_steps) > 0:
            total = int(self.config.max_steps)
        overrides: Dict[str, Any] = {"label_smoothing_factor": 0.0}
        if not len(self.eval_dataset):
            overrides["eval_strategy"] = "no"
        args = hft.build_training_arguments(self.config, self.config.effective_output_dir(),
                                            CrossEncoderTrainingArguments, **overrides)
        self._trainer = CrossEncoderTrainer(
            model=self.model, args=args, train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset if len(self.eval_dataset) else None,
            loss=BinaryCrossEntropyLoss(self.model),
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
        now = self.evaluate_now("nach dem Training")
        metrics.update(now)
        for k, v in self.baseline.items():
            metrics[f"{k}_before"] = v
        if "accuracy" not in now and "hit_at_1" in now:
            # Analyse zeigt "accuracy" prominent: bei Rangfolgen der Treffer auf Platz 1.
            metrics["accuracy"] = now["hit_at_1"]
        metrics["rerank_format"] = self.spec.get("format", "")
        if now and self.baseline:
            MessageProtocol.status("validating", "Vorher -> nachher: " + ", ".join(
                f"{k} {self.baseline[k]:.3f} -> {now[k]:.3f}" for k in now if k in self.baseline))
        return metrics

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        R.save_spec(out, {**self.spec, "task": "reranker"})
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)
