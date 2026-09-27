"""Test-Plugin: Sentence Embeddings.

single:
  "Satz A ||| Satz B"  -> Kosinus-Aehnlichkeit der beiden Saetze
  "Satz"               -> Dimension/Norm des Vektors und, wenn ein Dataset
                          gewaehlt ist (plugin_config.corpus_path), die
                          aehnlichsten Texte daraus
dataset:
  Paare/Tripel  -> Recall@1/@5/@10 und MRR@10: jede Frage sucht ihre Antwort
                   unter ALLEN Dokumenten des Test-Splits
  Score-Paare   -> Spearman zwischen Kosinus und Soll-Score
  Text/Klasse   -> 1-NN-Accuracy (aehnlichster anderer Text, gleiche Klasse?)
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TestConfig
from core.protocol import TestProtocol
from _shared_classify import resolve_device
from ft_data.media import sample as random_sample
from ft_data.splits import evaluation_files
from ft_data import embedding as emb

MAX_CORPUS = 2000


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.model = None
        self.spec: Dict[str, Any] = {}
        self.query_prefix = ""
        self.document_prefix = ""
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            from ft_data.deps import missing
            raise missing("sentence-transformers", what="Der Embedding-Test")
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modellpfad existiert nicht: {model_path}")
        TestProtocol.status("loading", "Lade Embedding-Modell...")
        device = resolve_device()
        self.model = SentenceTransformer(str(model_path), device=device.type)
        self.spec = emb.load_spec(model_path)
        cfg = self.config.plugin_config or {}
        self.query_prefix = str(cfg.get("query_prefix") or self.spec.get("query_prefix") or "")
        self.document_prefix = str(cfg.get("document_prefix") or self.spec.get("document_prefix") or "")
        dim = 0
        for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            fn = getattr(self.model, name, None)
            if callable(fn):
                dim = int(fn() or 0)
                break
        TestProtocol.status("loading", f"Modell geladen | Dimension: {dim or '?'} | Gerät: {device}")

    def _encode(self, texts: List[str], prefix: str = ""):
        return self.model.encode([prefix + t for t in texts], convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False,
                                 batch_size=max(int(self.config.batch_size or 16), 8))

    # ── Einzel-Eingabe ──────────────────────────────────────────────────────
    def run_single(self):
        import numpy as np

        text = (self.config.single_input or "").strip()
        if not text:
            raise ValueError("Eingabetext ist leer.")
        t0 = time.time()
        if "|||" in text:
            a, b = (p.strip() for p in text.split("|||", 1))
            if not a or not b:
                raise ValueError('Fuer einen Vergleich beide Saetze angeben: "Satz A ||| Satz B"')
            va = self._encode([a], self.query_prefix)[0]
            vb = self._encode([b], self.document_prefix)[0]
            sim = float(np.dot(va, vb))
            TestProtocol.complete_single(
                predicted=f"Kosinus-Aehnlichkeit: {sim:.3f}", confidence=None,
                top_predictions=[{"label": b, "score": sim}], inference_time=time.time() - t0,
            )
            return

        raw = self.model.encode([self.query_prefix + text], convert_to_numpy=True, show_progress_bar=False)[0]
        info = f"Vektor mit {raw.shape[0]} Dimensionen (Norm {float(np.linalg.norm(raw)):.3f})"
        top: List[Dict[str, Any]] = []
        corpus_path = (self.config.plugin_config or {}).get("corpus_path") or self.config.dataset_path
        if corpus_path and Path(corpus_path).exists():
            corpus = self._corpus(Path(corpus_path))
            if corpus:
                TestProtocol.status("running", f"Vergleiche mit {len(corpus)} Texten aus dem Dataset...")
                vecs = self._encode(corpus, self.document_prefix)
                q = raw / max(float(np.linalg.norm(raw)), 1e-12)
                idx, sims = emb.rank_corpus(q[None, :], vecs, top_k=5)
                top = [{"label": corpus[int(i)], "score": float(s)} for i, s in zip(idx[0], sims[0])]
                info += f" | aehnlichster Text ({top[0]['score']:.3f}): {top[0]['label']}"
        TestProtocol.complete_single(
            predicted=info, confidence=None, top_predictions=top, inference_time=time.time() - t0,
        )

    def _corpus(self, root: Path) -> List[str]:
        """Texte des gewaehlten Datasets (erst Test-, dann Val-, dann Train-Split)."""
        _, files = evaluation_files(root, emb.DATA_EXTS)
        rows: List[Dict[str, Any]] = []
        for f in files:
            rows.extend(emb.load_rows(f))
        if not rows:
            return []
        try:
            spec = emb.resolve_spec(list(rows[0].keys()), rows, self._spec_overrides())
        except ValueError:
            spec = {"text_column": next(iter(rows[0].keys()))}
        return emb.corpus_texts(rows, spec, MAX_CORPUS)

    def _spec_overrides(self) -> Dict[str, Any]:
        keys = ("format", "anchor_column", "positive_column", "negative_column",
                "score_column", "text_column", "label_column")
        out = {k: v for k, v in self.spec.items() if k in keys and v}
        out.update({k: v for k, v in (self.config.plugin_config or {}).items() if k in keys and v})
        return out

    # ── Dataset ─────────────────────────────────────────────────────────────
    def run_dataset(self):
        root = Path(self.config.dataset_path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
        split, files = evaluation_files(root, emb.DATA_EXTS)
        if not files:
            raise ValueError(f"Keine Datendatei in '{root}' gefunden (erwartet: {', '.join(emb.DATA_EXTS)}).")
        if split:
            TestProtocol.status("loading", f"Verwende Split: {split}")
        rows: List[Dict[str, Any]] = []
        for f in files:
            rows.extend(emb.load_rows(f))
        if not rows:
            raise ValueError("Dataset enthaelt keine Zeilen.")
        try:
            spec = emb.resolve_spec(list(rows[0].keys()), rows, self._spec_overrides())
        except ValueError:
            spec = emb.resolve_spec(list(rows[0].keys()), rows, self.config.plugin_config or {})
        TestProtocol.status("loading", f"Format: {emb.describe(spec)}")
        rows = random_sample(rows, self.config.max_samples)
        cols = emb.to_columns(rows, spec)  # Klassen als Text — so stehen sie lesbar im Ergebnis

        started = time.time()
        if spec["format"] in ("pairs", "triplets"):
            results, metrics, acc, correct = self._retrieval(cols)
        elif spec["format"] == "scored":
            results, metrics, acc, correct = self._scored(cols)
        else:
            results, metrics, acc, correct = self._labeled(cols)
        if results is None:
            TestProtocol.status("stopped", "Test abgebrochen.")
            return
        elapsed = max(time.time() - started, 1e-6)

        TestProtocol.status("running", " | ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
        out_dir = Path(self.config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_file = out_dir / "results.json"
        results_file.write_text(json.dumps({"predictions": results, "metrics": metrics},
                                           indent=2, ensure_ascii=False), encoding="utf-8")
        TestProtocol.complete_dataset(
            results_file=str(results_file),
            total_samples=len(results),
            accuracy=acc,
            correct=correct,
            average_loss=None,
            average_inference_time=elapsed / max(len(results), 1),
            samples_per_second=len(results) / elapsed,
            metrics=metrics,
        )

    def _retrieval(self, cols):
        queries, corpus, relevant = emb.retrieval_set(cols)
        if not queries:
            raise ValueError("Keine vollstaendigen Paare im Dataset.")
        q_ids, d_ids = list(queries), list(corpus)
        TestProtocol.status("running", f"{len(q_ids)} Fragen gegen {len(d_ids)} Dokumente...")
        TestProtocol.progress(current=0, total=len(q_ids))
        d_vec = self._encode([corpus[d] for d in d_ids], self.document_prefix)
        if self.is_stopped:
            return None, None, None, None
        q_vec = self._encode([queries[q] for q in q_ids], self.query_prefix)
        idx, sims = emb.rank_corpus(q_vec, d_vec, top_k=10)
        ranked = [[d_ids[int(i)] for i in row] for row in idx]
        rel = [relevant[q] for q in q_ids]
        metrics = emb.retrieval_metrics(ranked, rel)
        results = []
        correct = 0
        for n, q in enumerate(q_ids):
            hit = ranked[n][0] in relevant[q]
            correct += int(hit)
            results.append({
                "sample_id": n + 1,
                "input_text": queries[q][:500],
                "expected_output": corpus[next(iter(relevant[q]))][:500],
                "predicted_output": corpus[ranked[n][0]][:500],
                "is_correct": bool(hit),
                "confidence": float(sims[n][0]),
                "inference_time": 0.0,
                "top_predictions": [{"label": corpus[ranked[n][j]][:200], "score": float(sims[n][j])}
                                    for j in range(min(5, len(ranked[n])))],
            })
        TestProtocol.progress(current=len(q_ids), total=len(q_ids))
        return results, metrics, metrics.get("recall_at_1"), correct

    def _scored(self, cols):
        import numpy as np
        a = self._encode(cols["sentence1"], self.query_prefix)
        b = self._encode(cols["sentence2"], self.document_prefix)
        sims = (a * b).sum(axis=1)
        gold = list(cols["score"])
        metrics = {"spearman": emb.spearman(list(sims), gold),
                   "pearson": float(np.corrcoef(sims, gold)[0, 1]) if len(gold) > 1 and np.std(gold) > 0 else 0.0}
        results = []
        for i, (s1, s2, g, p) in enumerate(zip(cols["sentence1"], cols["sentence2"], gold, sims)):
            results.append({
                "sample_id": i + 1,
                "input_text": f"{s1} ||| {s2}"[:500],
                "expected_output": f"{g:.3f}",
                "predicted_output": f"{float(p):.3f}",
                # "richtig" = Kosinus liegt hoechstens 0,2 neben dem Soll (beides 0..1).
                "is_correct": bool(abs(float(p) - float(g)) <= 0.2),
                "confidence": None,
                "inference_time": 0.0,
            })
        TestProtocol.progress(current=len(results), total=len(results))
        # Keine "Treffer"-Quote: bei Scores zaehlt die Rangkorrelation.
        return results, metrics, None, None

    def _labeled(self, cols):
        texts, labels = cols["text"], cols["label"]
        if len(texts) < 2:
            raise ValueError("Zu wenige Texte fuer eine 1-NN-Auswertung.")
        vecs = self._encode(texts)
        import numpy as np
        sims = vecs @ vecs.T
        np.fill_diagonal(sims, -np.inf)
        nn = sims.argmax(axis=1)
        results, correct = [], 0
        for i, t in enumerate(texts):
            ok = labels[int(nn[i])] == labels[i]
            correct += int(ok)
            results.append({
                "sample_id": i + 1, "input_text": t[:500],
                "expected_output": str(labels[i]), "predicted_output": str(labels[int(nn[i])]),
                "is_correct": bool(ok), "confidence": float(sims[i, int(nn[i])]), "inference_time": 0.0,
            })
        acc = correct / len(texts)
        TestProtocol.progress(current=len(results), total=len(results))
        return results, {"knn_accuracy": acc}, acc, correct
