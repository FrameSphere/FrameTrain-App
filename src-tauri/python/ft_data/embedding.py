"""Sentence Embeddings: Datenformat erkennen, Retrieval-Metriken rechnen.

Gemeinsam fuer Training und Test. Vier Formate, jedes mit eigenem Loss:

  pairs     anchor/positive (query/document, question/answer, sentence1/sentence2
            ohne Score)                     -> MultipleNegativesRankingLoss
  triplets  anchor/positive/negative        -> MNRL mit harten Negativen
  scored    sentence1/sentence2/score       -> CoSENTLoss (Score auf 0..1)
  labeled   text/label                      -> BatchAllTripletLoss

Die gewaehlten Spalten stehen nach dem Training in SPEC_FILE neben dem Modell;
der Test liest sie von dort, statt neu zu raten.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATA_EXTS = (".jsonl", ".json", ".csv", ".tsv", ".parquet")
SPEC_FILE = "frametrain_embedding.json"

ANCHOR_COLUMNS = ["anchor", "query", "question", "sentence1", "sentence_a", "text1",
                  "premise", "sentence", "source", "q"]
POSITIVE_COLUMNS = ["positive", "document", "answer", "passage", "sentence2", "sentence_b",
                    "text2", "hypothesis", "pos", "target", "context", "doc"]
NEGATIVE_COLUMNS = ["negative", "hard_negative", "neg", "negative_passage", "negative_document"]
SCORE_COLUMNS = ["score", "similarity", "relatedness_score", "sim", "similarity_score", "label"]
TEXT_COLUMNS = ["text", "sentence", "content", "document"]
LABEL_COLUMNS = ["label", "labels", "category", "class", "intent", "topic"]

FORMAT_NAMES = {
    "pairs": "Paare (MultipleNegativesRankingLoss)",
    "triplets": "Tripel mit harten Negativen (MultipleNegativesRankingLoss)",
    "scored": "Paare mit Score (CoSENTLoss)",
    "labeled": "Texte mit Klasse (BatchAllTripletLoss)",
}


def _is_number(v: Any) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return not (isinstance(v, float) and math.isnan(v))
    try:
        float(str(v))
        return True
    except (TypeError, ValueError):
        return False


def _pick(cols: Sequence[str], candidates: Sequence[str], exclude=()) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        c = lower.get(cand)
        if c and c not in exclude:
            return c
    return None


def resolve_spec(columns: Sequence[str], rows: Sequence[Dict[str, Any]],
                 cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Erkennt das Format. Spaltennamen aus cfg haben Vorrang.

    cfg-Schluessel: format, anchor_column, positive_column, negative_column,
    score_column, text_column, label_column.
    """
    cfg = cfg or {}
    cols = [c for c in columns if not str(c).startswith("__")]
    sample = [r for r in rows[:200]]
    forced = str(cfg.get("format") or "").strip().lower() or None

    anchor = cfg.get("anchor_column") or _pick(cols, ANCHOR_COLUMNS)
    positive = cfg.get("positive_column") or _pick(cols, POSITIVE_COLUMNS, exclude=(anchor,))
    negative = cfg.get("negative_column") or _pick(cols, NEGATIVE_COLUMNS, exclude=(anchor, positive))
    score = cfg.get("score_column") or _pick(cols, SCORE_COLUMNS, exclude=(anchor, positive, negative))
    if score and not all(_is_number(r.get(score)) for r in sample if r.get(score) is not None):
        score = None  # "label" mit Text ist eine Klasse, kein Score

    for name, col in (("anchor_column", anchor), ("positive_column", positive),
                      ("negative_column", negative), ("score_column", score)):
        if cfg.get(name) and cfg[name] not in cols:
            raise ValueError(f"Spalte '{cfg[name]}' ({name}) fehlt. Vorhandene Spalten: {cols}")

    def spec(fmt: str, **kw) -> Dict[str, Any]:
        base = {"format": fmt, "anchor_column": None, "positive_column": None,
                "negative_column": None, "score_column": None, "text_column": None,
                "label_column": None, "score_scale": 1.0}
        base.update(kw)
        return base

    if anchor and positive and (forced in (None, "triplets")) and negative:
        return spec("triplets", anchor_column=anchor, positive_column=positive, negative_column=negative)
    if anchor and positive and (forced in (None, "scored")) and score:
        values = [float(r[score]) for r in rows if r.get(score) is not None and _is_number(r.get(score))]
        return spec("scored", anchor_column=anchor, positive_column=positive, score_column=score,
                    score_scale=score_scale(values))
    if anchor and positive and forced in (None, "pairs"):
        return spec("pairs", anchor_column=anchor, positive_column=positive)

    text = cfg.get("text_column") or _pick(cols, TEXT_COLUMNS)
    label = cfg.get("label_column") or _pick(cols, LABEL_COLUMNS, exclude=(text,))
    if text and label and text in cols and label in cols and forced in (None, "labeled"):
        return spec("labeled", text_column=text, label_column=label)

    raise ValueError(
        f"Datenformat fuer Embeddings nicht erkannt. Vorhandene Spalten: {cols}.\n"
        "Erwartet: anchor/positive (oder query/document, question/answer, sentence1/sentence2),\n"
        "optional negative oder score, oder text/label. Eigene Namen in der Plugin-Konfiguration, z.B.:\n"
        '  {"anchor_column": "frage", "positive_column": "antwort"}'
    )


def score_scale(values: Sequence[float]) -> float:
    """Teiler, der Scores auf 0..1 bringt: STS-B hat 0..5, andere 0..1 oder 0..100."""
    if not values:
        return 1.0
    hi = max(values)
    if hi <= 1.0:
        return 1.0
    if hi <= 5.0:
        return 5.0
    if hi <= 10.0:
        return 10.0
    return float(hi)


def describe(spec: Dict[str, Any]) -> str:
    fmt = spec.get("format")
    cols = {
        "pairs": f"'{spec.get('anchor_column')}' -> '{spec.get('positive_column')}'",
        "triplets": f"'{spec.get('anchor_column')}' -> '{spec.get('positive_column')}' / "
                    f"negativ '{spec.get('negative_column')}'",
        "scored": f"'{spec.get('anchor_column')}' ~ '{spec.get('positive_column')}' "
                  f"(Score '{spec.get('score_column')}' / {spec.get('score_scale')})",
        "labeled": f"'{spec.get('text_column')}' mit Klasse '{spec.get('label_column')}'",
    }.get(fmt, "")
    return f"{FORMAT_NAMES.get(fmt, fmt)}: {cols}"


def _txt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, (list, tuple)):
        return str(v[0]) if v else ""  # z.B. answers: ["..."]
    if isinstance(v, dict) and "text" in v:
        return _txt(v["text"])
    return str(v)


def to_columns(rows: Sequence[Dict[str, Any]], spec: Dict[str, Any],
               label2id: Optional[Dict[str, int]] = None) -> Dict[str, List[Any]]:
    """Zeilen -> Spalten in genau der Reihenfolge, die der Loss erwartet.

    Sentence-Transformers ordnet Spalten nach Position, nicht nach Namen zu —
    eine zusaetzliche id-Spalte vorne haette der Loss als Anker gelesen.
    """
    fmt = spec["format"]
    if fmt == "labeled":
        texts, labels = [], []
        for r in rows:
            t = _txt(r.get(spec["text_column"])).strip()
            lab = r.get(spec["label_column"])
            if not t or lab is None:
                continue
            key = str(lab)
            if label2id is not None:
                if key not in label2id:
                    label2id[key] = len(label2id)
                labels.append(label2id[key])
            else:
                labels.append(key)
            texts.append(t)
        return {"text": texts, "label": labels}

    a_col, p_col = spec["anchor_column"], spec["positive_column"]
    out: Dict[str, List[Any]] = {"anchor": [], "positive": []}
    if fmt == "triplets":
        out["negative"] = []
    if fmt == "scored":
        out = {"sentence1": [], "sentence2": [], "score": []}
    scale = float(spec.get("score_scale") or 1.0)
    for r in rows:
        a = _txt(r.get(a_col)).strip()
        p = _txt(r.get(p_col)).strip()
        if not a or not p:
            continue
        if fmt == "scored":
            s = r.get(spec["score_column"])
            if s is None or not _is_number(s):
                continue
            out["sentence1"].append(a)
            out["sentence2"].append(p)
            out["score"].append(max(0.0, min(1.0, float(s) / scale)))
            continue
        if fmt == "triplets":
            n = _txt(r.get(spec["negative_column"])).strip()
            if not n:
                continue
            out["negative"].append(n)
        out["anchor"].append(a)
        out["positive"].append(p)
    return out


def retrieval_set(cols: Dict[str, List[Any]]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, set]]:
    """Paare/Tripel -> (queries, corpus, relevante Dokumente) fuer Recall@k.

    Gleiche Texte werden zu einem Dokument zusammengefasst — sonst traefe eine
    Frage "ihr" Duplikat und gaelte trotzdem als Fehlgriff.
    """
    queries: Dict[str, str] = {}
    corpus: Dict[str, str] = {}
    relevant: Dict[str, set] = {}
    doc_id: Dict[str, str] = {}
    q_id: Dict[str, str] = {}

    def did(text: str) -> str:
        if text not in doc_id:
            doc_id[text] = f"d{len(doc_id)}"
            corpus[doc_id[text]] = text
        return doc_id[text]

    negs = cols.get("negative") or []
    for i, (a, p) in enumerate(zip(cols.get("anchor", []), cols.get("positive", []))):
        if a not in q_id:
            q_id[a] = f"q{len(q_id)}"
            queries[q_id[a]] = a
        relevant.setdefault(q_id[a], set()).add(did(p))
        if i < len(negs) and negs[i]:
            did(negs[i])
    return queries, corpus, relevant


def rank_corpus(query_emb, corpus_emb, top_k: int = 10):
    """Kosinus-Aehnlichkeit (normierte Vektoren) -> Indizes und Scores der besten k."""
    import numpy as np
    q = np.asarray(query_emb, dtype="float32")
    c = np.asarray(corpus_emb, dtype="float32")
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    c = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
    sims = q @ c.T
    k = min(top_k, sims.shape[1])
    idx = np.argsort(-sims, axis=1)[:, :k]
    return idx, np.take_along_axis(sims, idx, axis=1)


def retrieval_metrics(ranked_ids: Sequence[Sequence[str]], relevant: Sequence[set]) -> Dict[str, float]:
    """Recall@1/@5/@10 (Anteil der Fragen mit Treffer unter den ersten k) und MRR@10."""
    n = len(ranked_ids)
    if n == 0:
        return {}
    hits = {1: 0, 5: 0, 10: 0}
    mrr = 0.0
    for ids, rel in zip(ranked_ids, relevant):
        for k in hits:
            if any(d in rel for d in list(ids)[:k]):
                hits[k] += 1
        for rank, d in enumerate(list(ids)[:10], start=1):
            if d in rel:
                mrr += 1.0 / rank
                break
    return {"recall_at_1": hits[1] / n, "recall_at_5": hits[5] / n,
            "recall_at_10": hits[10] / n, "mrr_at_10": mrr / n}


def knn_label_accuracy(embeddings, labels: Sequence[Any]) -> float:
    """Leave-one-out 1-NN: hat der aehnlichste andere Text dieselbe Klasse?"""
    import numpy as np
    n = len(labels)
    if n < 2:
        return 0.0
    e = np.asarray(embeddings, dtype="float32")
    e = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None)
    sims = e @ e.T
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(axis=1)
    return float(sum(1 for i in range(n) if labels[nn[i]] == labels[i]) / n)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rangkorrelation ohne scipy-Pflicht (scipy ist aber meist da)."""
    try:
        from scipy.stats import spearmanr
        v = spearmanr(a, b).correlation
        return float(v) if v == v else 0.0
    except ImportError:
        import numpy as np
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        if ra.std() == 0 or rb.std() == 0:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])


def corpus_texts(rows: Sequence[Dict[str, Any]], spec: Dict[str, Any], limit: int = 2000) -> List[str]:
    """Alle Texte eines Datasets (fuer "aehnlichste Texte" im Einzeltest), ohne Dubletten."""
    keys = [spec.get(k) for k in ("positive_column", "negative_column", "anchor_column", "text_column")]
    seen: Dict[str, None] = {}
    for r in rows:
        for k in keys:
            if k:
                t = _txt(r.get(k)).strip()
                if t and t not in seen:
                    seen[t] = None
                    if len(seen) >= limit:
                        return list(seen)
    return list(seen)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Tabelle -> Zeilen (json/jsonl/csv/tsv/parquet)."""
    ext = Path(path).suffix.lower()
    if ext in (".json", ".jsonl"):
        from .tokens import read_table
        return read_table(Path(path))[0]
    import pandas as pd
    if ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",")
    return df.to_dict(orient="records")


def save_spec(model_dir: Path, spec: Dict[str, Any]) -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / SPEC_FILE).write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")


def load_spec(model_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((Path(model_dir) / SPEC_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
