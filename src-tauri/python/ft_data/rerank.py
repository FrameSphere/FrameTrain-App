"""
ft_data.rerank – Reranker / Cross-Encoder (Training und Test).

Ein Cross-Encoder liest Frage und Passage gemeinsam und gibt EINE Zahl aus:
wie gut passt die Passage? Gemeinsam fuer das Train- und das Test-Plugin
`reranker`.

Erkannte Formate
  scored     query + passage + score/label     (0/1, 0..1 oder 0..5 – wird auf 0..1 gebracht)
             auch sentence1/sentence2/label, question/document/relevance …
  triplets   query + positive [+ negative | negatives (Liste)]
             Fehlen Negative, werden Passagen anderer Fragen als Negative gezogen.

Training: alles wird zu (query, passage, label 0..1) -> BinaryCrossEntropyLoss.

Bewertung je Frage ueber ihre Kandidaten (Positive, Negative und bei
Tripeln zusaetzlich Passagen anderer Fragen):
  MRR@10, nDCG@10 (abgestufte Relevanz = Label), Hit@1
  bei Paaren mit Score zusaetzlich Accuracy (Schwelle 0,5) und Spearman.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATA_EXTS = (".jsonl", ".json", ".csv", ".tsv", ".parquet")
SPEC_FILE = "frametrain_rerank.json"

QUERY_COLUMNS = ("query", "question", "anchor", "sentence1", "text1", "frage", "suchanfrage", "q")
PASSAGE_COLUMNS = ("passage", "document", "doc", "sentence2", "text2", "context", "answer", "text", "absatz")
POSITIVE_COLUMNS = ("positive", "pos", "positive_passage", "relevant")
NEGATIVE_COLUMNS = ("negative", "neg", "negative_passage")
NEGATIVES_COLUMNS = ("negatives", "hard_negatives", "negs")
POSITIVES_COLUMNS = ("positives", "pos_list")
SCORE_COLUMNS = ("score", "label", "relevance", "rel", "relevant_score", "similarity")


def _pick(cols: Sequence[str], candidates: Sequence[str], exclude=()) -> Optional[str]:
    lower = {c.lower(): c for c in cols if c not in exclude}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _txt(v: Any) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v).strip()


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                return _as_list(json.loads(s))
            except ValueError:
                pass
        return [s] if s else []
    if isinstance(v, (list, tuple)):
        return [_txt(x) for x in v if _txt(x)]
    return [_txt(v)] if _txt(v) else []


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return float(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def resolve_spec(columns: Sequence[str], rows: Sequence[Dict[str, Any]],
                 overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ov = overrides or {}
    cols = list(columns)

    def given(key):
        name = str(ov.get(key) or "").strip()
        if name and name not in cols:
            raise ValueError(f"Spalte '{name}' ({key}) nicht gefunden. Vorhanden: {sorted(cols)}")
        return name or None

    q = given("query_column") or _pick(cols, QUERY_COLUMNS)
    if not q:
        raise ValueError(
            f"Keine Frage-Spalte gefunden. Vorhanden: {sorted(cols)}. Erwartet: query + passage + "
            "score/label oder query + positive [+ negative]. Eigene Namen: plugin_config.query_column, "
            "passage_column, score_column, positive_column, negative_column.")
    pos = given("positive_column") or _pick(cols, POSITIVE_COLUMNS + POSITIVES_COLUMNS, exclude=(q,))
    if pos and not given("passage_column"):
        neg = given("negative_column") or _pick(cols, NEGATIVE_COLUMNS + NEGATIVES_COLUMNS, exclude=(q, pos))
        return {"format": "triplets", "query_column": q, "positive_column": pos, "negative_column": neg}
    passage = given("passage_column") or _pick(cols, PASSAGE_COLUMNS, exclude=(q,))
    score = given("score_column") or _pick(cols, SCORE_COLUMNS, exclude=(q, passage))
    if not passage or not score:
        raise ValueError(
            f"Format nicht erkannt (Frage-Spalte '{q}', Spalten {sorted(cols)}). Erwartet: "
            "query + passage + score/label oder query + positive [+ negative].")
    values = [_num(r.get(score)) for r in rows[:2000]]
    values = [v for v in values if v is not None]
    if not values:
        raise ValueError(f"Spalte '{score}' enthaelt keine Zahlen (erwartet 0/1 oder einen Score).")
    return {"format": "scored", "query_column": q, "passage_column": passage, "score_column": score,
            "score_scale": score_scale(values), "score_min": min(0.0, min(values))}


def score_scale(values: Sequence[float]) -> float:
    hi = max(values) if values else 1.0
    if hi <= 1.0:
        return 1.0
    for cand in (5.0, 10.0, 100.0):
        if hi <= cand:
            return cand
    return float(hi)


def describe(spec: Dict[str, Any]) -> str:
    if spec["format"] == "triplets":
        neg = spec.get("negative_column")
        return (f"Frage + positive Passage ({spec['query_column']} / {spec['positive_column']}"
                + (f" / {neg}" if neg else ", Negative aus anderen Fragen") + ")")
    return (f"Paare mit Relevanz ({spec['query_column']} / {spec['passage_column']} / "
            f"{spec['score_column']}, Skala 0..{spec['score_scale']:g})")


def groups(rows: Sequence[Dict[str, Any]], spec: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """{query: {passage: relevanz 0..1}} — Grundlage fuer Training und Bewertung."""
    out: Dict[str, Dict[str, float]] = {}
    if spec["format"] == "scored":
        scale, lo = float(spec.get("score_scale") or 1.0), float(spec.get("score_min") or 0.0)
        for r in rows:
            q, p, s = _txt(r.get(spec["query_column"])), _txt(r.get(spec["passage_column"])), _num(r.get(spec["score_column"]))
            if not q or not p or s is None:
                continue
            rel = max(0.0, min(1.0, (s - lo) / max(scale - lo, 1e-9)))
            g = out.setdefault(q, {})
            g[p] = max(g.get(p, 0.0), rel)
        return out
    for r in rows:
        q = _txt(r.get(spec["query_column"]))
        if not q:
            continue
        g = out.setdefault(q, {})
        for p in _as_list(r.get(spec["positive_column"])):
            g[p] = 1.0
        if spec.get("negative_column"):
            for n in _as_list(r.get(spec["negative_column"])):
                g.setdefault(n, 0.0)
    return {q: g for q, g in out.items() if g}


def add_sampled_negatives(grp: Dict[str, Dict[str, float]], per_query: int, seed: int = 42
                          ) -> Dict[str, Dict[str, float]]:
    """Fragen mit zu wenigen Negativen bekommen Passagen anderer Fragen dazu."""
    rng = random.Random(seed)
    pool = sorted({p for g in grp.values() for p in g})
    out = {}
    for q, g in grp.items():
        g = dict(g)
        have = sum(1 for v in g.values() if v < 0.5)
        need = max(0, per_query - have)
        if need and len(pool) > len(g):
            cands = [p for p in rng.sample(pool, min(len(pool), need + len(g) + 5)) if p not in g]
            for p in cands[:need]:
                g[p] = 0.0
        out[q] = g
    return out


def to_pairs(grp: Dict[str, Dict[str, float]]) -> Dict[str, List[Any]]:
    """{query, passage, label} als Spalten (fuer datasets.Dataset)."""
    cols: Dict[str, List[Any]] = {"query": [], "passage": [], "label": []}
    for q, g in grp.items():
        for p, rel in g.items():
            cols["query"].append(q); cols["passage"].append(p); cols["label"].append(float(rel))
    return cols


def split_groups(grp: Dict[str, Dict[str, float]], fraction: float, seed: int
                 ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Validierung nach Fragen trennen — sonst saehe das Modell dieselbe Frage im Training."""
    qs = sorted(grp)
    random.Random(seed).shuffle(qs)
    n_val = max(1, int(round(len(qs) * fraction))) if len(qs) >= 10 else 0
    val = {q: grp[q] for q in qs[:n_val]}
    train = {q: grp[q] for q in qs[n_val:]}
    return train, val


# ─── Kennzahlen ───────────────────────────────────────────────────────────────

def mrr_at_k(ranked_rels: Sequence[float], k: int = 10) -> float:
    for i, r in enumerate(ranked_rels[:k]):
        if r >= 0.5:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_rels: Sequence[float], k: int = 10) -> float:
    def dcg(rels):
        return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = dcg(sorted(ranked_rels, reverse=True))
    return dcg(list(ranked_rels)) / ideal if ideal > 0 else 0.0


def rank(grp_scores: Dict[str, List[Tuple[str, float, float]]]) -> Dict[str, List[Tuple[str, float, float]]]:
    """{query: [(passage, score, relevanz)]} absteigend nach Modell-Score."""
    return {q: sorted(items, key=lambda x: x[1], reverse=True) for q, items in grp_scores.items()}


def ranking_metrics(ranked: Dict[str, List[Tuple[str, float, float]]], k: int = 10) -> Dict[str, float]:
    """Nur Fragen mit mindestens einer relevanten Passage und >= 2 Kandidaten zaehlen."""
    mrr, ndcg, hit = [], [], []
    for items in ranked.values():
        rels = [r for _, _, r in items]
        if len(rels) < 2 or not any(r >= 0.5 for r in rels):
            continue
        mrr.append(mrr_at_k(rels, k))
        ndcg.append(ndcg_at_k(rels, k))
        hit.append(float(rels[0] >= 0.5))
    if not mrr:
        return {}
    n = len(mrr)
    return {f"mrr_at_{k}": sum(mrr) / n, f"ndcg_at_{k}": sum(ndcg) / n, "hit_at_1": sum(hit) / n,
            "ranked_queries": float(n)}


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    from .embedding import spearman as sp
    return sp(list(a), list(b))


def pair_metrics(scores: Sequence[float], labels: Sequence[float]) -> Dict[str, float]:
    """Accuracy (Score und Label beide ueber/unter 0,5) und Spearman."""
    if not scores:
        return {}
    acc = sum(1 for s, l in zip(scores, labels) if (s >= 0.5) == (l >= 0.5)) / len(scores)
    out = {"accuracy": acc}
    if len(set(labels)) > 1:
        out["spearman"] = spearman(scores, labels)
    return out


def evaluate(score_fn, grp: Dict[str, Dict[str, float]], fmt: str) -> Tuple[Dict[str, float], Dict[str, List]]:
    """score_fn(list[(q, p)]) -> list[float]. Gibt (Kennzahlen, Rangfolge) zurueck."""
    pairs = [(q, p, rel) for q, g in grp.items() for p, rel in g.items()]
    if not pairs:
        return {}, {}
    scores = list(score_fn([(q, p) for q, p, _ in pairs]))
    by_q: Dict[str, List[Tuple[str, float, float]]] = {}
    for (q, p, rel), s in zip(pairs, scores):
        by_q.setdefault(q, []).append((p, float(s), rel))
    ranked = rank(by_q)
    m = ranking_metrics(ranked)
    if fmt == "scored":
        m.update(pair_metrics([float(s) for s in scores], [rel for _, _, rel in pairs]))
    return m, ranked


# ─── Dateien / Spec ───────────────────────────────────────────────────────────

def load_files(files) -> List[Dict[str, Any]]:
    from .embedding import load_rows
    rows: List[Dict[str, Any]] = []
    for f in files:
        rows.extend(load_rows(f))
    return rows


def save_spec(model_dir: Path, spec: Dict[str, Any]) -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / SPEC_FILE).write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")


def load_spec(model_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((Path(model_dir) / SPEC_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def parse_single_input(text: str) -> Tuple[str, List[str]]:
    """'Frage ||| Passage 1 ||| Passage 2' -> (frage, [passagen]). Ohne ||| nur die Frage."""
    parts = [p.strip() for p in (text or "").split("|||")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("Eingabe ist leer.")
    return parts[0], parts[1:]


def single_head(model_dir: Path) -> bool:
    """Hat das Modell schon einen Kopf mit genau einer Ausgabe (fertiger Reranker)?"""
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    archs = cfg.get("architectures") or []
    labels = cfg.get("id2label") or {}
    n = cfg.get("num_labels") or (len(labels) if labels else None)
    return any(str(a).endswith("ForSequenceClassification") for a in archs) and n == 1
