"""Test-Plugin: Reranker / Cross-Encoder.

single:
  "Frage ||| Passage A ||| Passage B"  -> Passagen nach Relevanz geordnet
  "Frage ||| Passage"                  -> Relevanz dieser Passage
  "Frage"                              -> ordnet Passagen aus dem gewaehlten Dataset
                                          (plugin_config.corpus_path)
dataset:
  je Frage ihre Kandidaten ordnen -> MRR@10, nDCG@10, Hit@1;
  bei Paaren mit Score zusaetzlich Accuracy und Spearman.
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol
from _shared_classify import resolve_device
from ft_data.splits import evaluation_files
from ft_data import rerank as R

MAX_CORPUS = 300
EVAL_NEGATIVES = 9


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.pc = config.plugin_config or {}
        self.model = None
        self.spec: Dict[str, Any] = {}
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError:
            from ft_data.deps import missing
            raise missing("sentence-transformers", what="Der Reranker-Test")
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")
        TestProtocol.status("loading", "Lade Cross-Encoder...")
        device = resolve_device()
        kwargs: Dict[str, Any] = {"device": device.type}
        if not R.single_head(model_path):
            TestProtocol.warning("Das Modell hat keinen Reranker-Kopf mit einer Ausgabe — die Scores sind ohne Training zufaellig.")
            kwargs["num_labels"] = 1
        self.model = CrossEncoder(str(model_path), **kwargs)
        self.spec = R.load_spec(model_path)
        TestProtocol.status("loading", f"Modell geladen | Gerät: {device}")

    def _score(self, pairs: List[Any]) -> List[float]:
        if not pairs:
            return []
        out = self.model.predict(pairs, batch_size=max(int(self.config.batch_size or 16), 8),
                                 show_progress_bar=False, convert_to_numpy=True)
        return [float(x) for x in out]

    def _overrides(self) -> Dict[str, Any]:
        keys = ("query_column", "passage_column", "score_column", "positive_column", "negative_column")
        return {k: v for k, v in self.pc.items() if k in keys and v}

    # ── Einzel-Eingabe ──────────────────────────────────────────────────────
    def run_single(self):
        t0 = time.time()
        query, passages = R.parse_single_input(self.config.single_input or "")
        if not passages:
            corpus = self.pc.get("corpus_path") or self.config.dataset_path
            if corpus and Path(corpus).exists():
                passages = self._corpus(Path(corpus))
            if not passages:
                raise ValueError('Passagen fehlen: "Frage ||| Passage A ||| Passage B" eingeben oder ein Dataset waehlen.')
            TestProtocol.status("running", f"Ordne {len(passages)} Passagen aus dem Dataset...")
        scores = self._score([(query, p) for p in passages])
        order = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
        top = [{"label": passages[i][:300], "score": scores[i]} for i in order[:10]]
        best = order[0]
        predicted = (f"Relevanz: {scores[best]:.3f}" if len(passages) == 1
                     else f"Platz 1 ({scores[best]:.3f}): {passages[best][:500]}")
        TestProtocol.complete_single(predicted=predicted, confidence=None, top_predictions=top,
                                     inference_time=time.time() - t0)

    def _corpus(self, root: Path) -> List[str]:
        _, files = evaluation_files(root, R.DATA_EXTS)
        rows = R.load_files(files)
        if not rows:
            return []
        try:
            spec = R.resolve_spec(list(rows[0].keys()), rows, self._overrides())
        except ValueError:
            return []
        seen: Dict[str, None] = {}
        for g in R.groups(rows, spec).values():
            for p in g:
                seen.setdefault(p, None)
                if len(seen) >= MAX_CORPUS:
                    return list(seen)
        return list(seen)

    # ── Dataset ─────────────────────────────────────────────────────────────
    def run_dataset(self):
        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        split, files = evaluation_files(root, R.DATA_EXTS)
        if not files:
            raise ValueError(f"Keine Datendatei in '{root}' gefunden (erwartet: {', '.join(R.DATA_EXTS)}).")
        if split:
            TestProtocol.status("loading", f"Verwende Split: {split}")
        rows = R.load_files(files)
        if not rows:
            raise ValueError("Dataset enthaelt keine Zeilen.")
        spec = R.resolve_spec(list(rows[0].keys()), rows, self._overrides())
        TestProtocol.status("loading", f"Format: {R.describe(spec)}")
        grp = R.groups(rows, spec)
        n = self.config.max_samples
        if n and len(grp) > int(n):
            import random
            keep = random.Random(42).sample(sorted(grp), int(n))
            grp = {q: grp[q] for q in keep}
        if spec["format"] == "triplets":
            grp = R.add_sampled_negatives(grp, EVAL_NEGATIVES, 43)
        total = len(grp)
        TestProtocol.progress(current=0, total=total)
        started = time.time()
        metrics, ranked = R.evaluate(self._score, grp, spec["format"])
        if self.is_stopped:
            TestProtocol.status("stopped", "Test abgebrochen.")
            return
        n_ranked = int(metrics.pop("ranked_queries", 0) or 0)
        results, correct = [], 0
        for i, (q, items) in enumerate(ranked.items(), start=1):
            rel_best = next((p for p, _, r in sorted(items, key=lambda x: -x[2]) if r >= 0.5), None)
            hit = bool(items and items[0][2] >= 0.5)
            correct += int(hit)
            results.append({
                "sample_id": i, "input_text": q[:500],
                "expected_output": (rel_best or "(keine relevante Passage)")[:500],
                "predicted_output": items[0][0][:500] if items else "",
                "is_correct": hit if rel_best else None,
                "confidence": items[0][1] if items else None,
                "inference_time": 0.0,
                "top_predictions": [{"label": p[:200], "score": s, "relevance": r} for p, s, r in items[:5]],
            })
        TestProtocol.progress(current=total, total=total)
        elapsed = max(time.time() - started, 1e-6)
        TestProtocol.status("running", " | ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
        out_dir = Path(self.config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_file = out_dir / "results.json"
        results_file.write_text(json.dumps({"predictions": results, "metrics": metrics}, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        acc = metrics.get("accuracy", metrics.get("hit_at_1"))
        TestProtocol.complete_dataset(
            results_file=str(results_file), total_samples=len(results), accuracy=acc,
            correct=correct if "accuracy" not in metrics and n_ranked else None,
            average_loss=None, average_inference_time=elapsed / max(len(results), 1),
            samples_per_second=len(results) / elapsed, metrics=metrics,
        )
