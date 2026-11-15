"""
ft_data.qa – Extraktives Frage-Antwort (SQuAD-Stil) fuer Training und Test.

Gemeinsam fuer das Train- und das Test-Plugin `question_answering`, damit
beide dieselben Daten, dieselben Fenster und dieselbe Bewertung sehen.

Erkannte Daten
  SQuAD-JSON     {"data": [{"paragraphs": [{"context": ..., "qas": [
                    {"id", "question", "answers": [{"text", "answer_start"}],
                     "is_impossible"}]}]}]}            (v1.1 und v2.0)
  HF-SQuAD       Zeilen mit context, question, answers={"text": [...], "answer_start": [...]}
                 (so liegen squad / squad_v2 als Parquet auf Hugging Face)
  flach          context, question, answer [, answer_start]  (CSV/TSV/JSONL/JSON/Parquet)
                 answers auch als Liste von Dicts oder Strings.
  Fehlt answer_start, wird die Antwort im Kontext gesucht (erst exakt, dann
  ohne Gross-/Kleinschreibung). Eine leere Antwort oder is_impossible=true ist
  "keine Antwort im Text" (SQuAD v2).

Lange Kontexte werden in Fenster geteilt (max_length, doc_stride Tokens
Ueberlappung). Ein Fenster ohne die Antwort lernt "keine Antwort" (CLS).

Bewertung wie das offizielle SQuAD-Skript: Exact Match und Token-F1 nach
Normalisierung (Kleinschreibung, ohne Satzzeichen und Artikel a/an/the),
das Maximum ueber alle Referenzantworten; bei "keine Antwort" zaehlt nur eine
leere Vorhersage als richtig.
"""
from __future__ import annotations

import collections
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DATA_EXTS = (".json", ".jsonl", ".csv", ".tsv", ".parquet")
SPEC_FILE = "frametrain_qa.json"

CONTEXT_COLUMNS = ("context", "passage", "paragraph", "document", "text", "kontext", "absatz")
QUESTION_COLUMNS = ("question", "query", "frage", "q")
ANSWERS_COLUMNS = ("answers", "antworten")
ANSWER_COLUMNS = ("answer", "answer_text", "antwort", "answer_span", "a")
START_COLUMNS = ("answer_start", "start", "start_char", "answer_start_char")
IMPOSSIBLE_COLUMNS = ("is_impossible", "impossible", "no_answer", "unanswerable")
ID_COLUMNS = ("id", "qid", "question_id")


# ─── Einlesen ─────────────────────────────────────────────────────────────────

def is_squad_json(data: Any) -> bool:
    """Das verschachtelte Original-Format (data -> paragraphs -> qas)?"""
    if isinstance(data, dict):
        data = data.get("data")
    return (isinstance(data, list) and bool(data) and isinstance(data[0], dict)
            and isinstance(data[0].get("paragraphs"), list))


def flatten_squad(data: Any) -> List[Dict[str, Any]]:
    """SQuAD v1.1/v2.0 -> eine Zeile pro Frage im HF-Format."""
    if isinstance(data, dict):
        data = data.get("data", [])
    rows: List[Dict[str, Any]] = []
    for article in data or []:
        for para in article.get("paragraphs", []) or []:
            context = str(para.get("context", ""))
            for qa in para.get("qas", []) or []:
                answers = qa.get("answers") or []
                if qa.get("is_impossible") and not answers:
                    answers = []
                rows.append({
                    "id": str(qa.get("id", "")),
                    "context": context,
                    "question": str(qa.get("question", "")),
                    "answers": {
                        "text": [str(a.get("text", "")) for a in answers if isinstance(a, dict)],
                        "answer_start": [int(a.get("answer_start", -1)) for a in answers if isinstance(a, dict)],
                    },
                    "is_impossible": bool(qa.get("is_impossible", False)),
                })
    return rows


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Eine Datei -> Zeilen. SQuAD-JSON wird flachgeklopft."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            data = None
        if data is not None and is_squad_json(data):
            return flatten_squad(data)
    from .embedding import load_rows as load_table
    return load_table(path)


def load_files(files: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for f in files:
        rows.extend(load_rows(f))
    return rows


# ─── Spalten & Normalisierung ─────────────────────────────────────────────────

def _pick(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def resolve_columns(columns: Sequence[str], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
    """Welche Spalten tragen Kontext, Frage und Antwort(en)?"""
    ov = overrides or {}
    cols = list(columns)

    def chosen(key: str, candidates: Sequence[str]) -> Optional[str]:
        name = str(ov.get(key) or "").strip()
        if name:
            if name not in cols:
                raise ValueError(f"Spalte '{name}' ({key}) nicht gefunden. Vorhanden: {sorted(cols)}")
            return name
        return _pick(cols, candidates)

    spec = {
        "context": chosen("context_column", CONTEXT_COLUMNS),
        "question": chosen("question_column", QUESTION_COLUMNS),
        "answers": _pick(cols, ANSWERS_COLUMNS),
        "answer": chosen("answer_column", ANSWER_COLUMNS),
        "answer_start": chosen("answer_start_column", START_COLUMNS),
        "impossible": _pick(cols, IMPOSSIBLE_COLUMNS),
        "id": _pick(cols, ID_COLUMNS),
    }
    if ov.get("answer_column"):
        spec["answers"] = None
    if not spec["context"] or not spec["question"] or not (spec["answers"] or spec["answer"]):
        raise ValueError(
            "Konnte das Frage-Antwort-Format nicht erkennen. Gefundene Spalten: "
            f"{sorted(cols)}.\nErwartet: context + question + answer (optional answer_start) "
            "oder answers im SQuAD-Format. Andere Namen lassen sich in plugin_config setzen: "
            "context_column, question_column, answer_column, answer_start_column."
        )
    return spec


def _txt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    return str(v)


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        if s.startswith(("[", "{")):
            try:
                return _as_list(json.loads(s))
            except ValueError:
                return [v]
        return [v] if s else []
    if hasattr(v, "tolist"):          # numpy-Array aus Parquet
        v = v.tolist()
    if isinstance(v, dict):
        return [v]
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _parse_answers(raw: Any) -> List[Tuple[str, Optional[int]]]:
    """Alle Formen von 'answers' -> [(text, start|None)]."""
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            raw = json.loads(raw)
        except ValueError:
            pass
    if isinstance(raw, dict) and "text" in raw:
        texts = _as_list(raw.get("text"))
        starts = _as_list(raw.get("answer_start"))
        out = []
        for i, t in enumerate(texts):
            s = starts[i] if i < len(starts) else None
            out.append((_txt(t), int(s) if s is not None and str(s).lstrip("-").isdigit() else None))
        return out
    out = []
    for item in _as_list(raw):
        if isinstance(item, dict):
            s = item.get("answer_start", item.get("start"))
            out.append((_txt(item.get("text", item.get("answer"))),
                        int(s) if s is not None and str(s).lstrip("-").isdigit() else None))
        else:
            out.append((_txt(item), None))
    return out


def find_answer(context: str, answer: str, hint: Optional[int] = None) -> int:
    """Startposition der Antwort im Kontext (-1 = nicht gefunden).

    Ein passender Hinweis (answer_start) gewinnt. Sonst das Vorkommen, das dem
    Hinweis am naechsten liegt, sonst das erste — erst exakt, dann ohne
    Gross-/Kleinschreibung.
    """
    if not answer:
        return -1
    if hint is not None and 0 <= hint and context[hint:hint + len(answer)] == answer:
        return hint
    for hay, needle in ((context, answer), (context.lower(), answer.lower())):
        hits = [m.start() for m in re.finditer(re.escape(needle), hay)]
        if hits:
            if hint is not None and hint >= 0:
                return min(hits, key=lambda h: abs(h - hint))
            return hits[0]
    return -1


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "ja", "y")


def normalize_rows(rows: Sequence[Dict[str, Any]], overrides: Optional[Dict[str, Any]] = None
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Zeilen -> {"id", "context", "question", "answers": {"text", "answer_start"}}.

    Antworten, die im Kontext nicht vorkommen, werden verworfen (gezaehlt in
    stats["not_found"]). Eine Zeile ohne verwertbare Antwort und ohne
    'keine Antwort'-Markierung faellt weg.
    """
    stats = {"rows": len(rows), "kept": 0, "no_answer": 0, "not_found": 0,
             "start_searched": 0, "empty": 0}
    if not rows:
        return [], stats
    spec = resolve_columns(list(rows[0].keys()), overrides)
    out: List[Dict[str, Any]] = []
    for n, row in enumerate(rows):
        context = _txt(row.get(spec["context"]))
        question = _txt(row.get(spec["question"])).strip()
        if not context.strip() or not question:
            stats["empty"] += 1
            continue
        if spec["answers"]:
            cands = _parse_answers(row.get(spec["answers"]))
        else:
            start = row.get(spec["answer_start"]) if spec["answer_start"] else None
            try:
                start = int(start) if start is not None and _txt(start).strip() != "" else None
            except (TypeError, ValueError):
                start = None
            cands = [(_txt(row.get(spec["answer"])), start)]
        impossible = bool(spec["impossible"] and _truthy(row.get(spec["impossible"])))
        texts, starts = [], []
        missing = 0
        for text, hint in cands:
            if not text.strip():
                continue
            pos = find_answer(context, text, hint)
            if pos < 0:
                missing += 1
                continue
            if hint is None or hint != pos:
                stats["start_searched"] += 1
            texts.append(context[pos:pos + len(text)])
            starts.append(pos)
        if missing and not texts:
            stats["not_found"] += 1
            continue
        if not texts:
            # leere Antwort, is_impossible oder answers=[] -> keine Antwort im Text
            impossible = True
        if impossible:
            texts, starts = [], []
            stats["no_answer"] += 1
        rid = _txt(row.get(spec["id"])) if spec["id"] else ""
        out.append({"id": rid or f"q{n}", "context": context, "question": question,
                    "answers": {"text": texts, "answer_start": starts}})
    # IDs muessen eindeutig sein — sonst vermischt die Nachbearbeitung Fragen.
    seen = collections.Counter(r["id"] for r in out)
    if any(c > 1 for c in seen.values()):
        for i, r in enumerate(out):
            r["id"] = f"{r['id']}#{i}"
    stats["kept"] = len(out)
    return out, stats


def describe(stats: Dict[str, int]) -> str:
    parts = [f"{stats['kept']} Fragen"]
    if stats.get("no_answer"):
        parts.append(f"{stats['no_answer']} ohne Antwort im Text (SQuAD v2)")
    if stats.get("start_searched"):
        parts.append(f"bei {stats['start_searched']} Antworten Position im Kontext gesucht")
    if stats.get("not_found"):
        parts.append(f"{stats['not_found']} verworfen (Antwort steht nicht im Kontext)")
    return " | ".join(parts)


# ─── Tokenisierung mit Fenstern ───────────────────────────────────────────────

def _pad_right(tokenizer) -> bool:
    return getattr(tokenizer, "padding_side", "right") == "right"


def _encode(tokenizer, questions, contexts, max_length: int, stride: int):
    right = _pad_right(tokenizer)
    stride = max(0, min(int(stride), max_length // 2))
    return tokenizer(
        [q.lstrip() for q in questions] if right else contexts,
        contexts if right else [q.lstrip() for q in questions],
        truncation="only_second" if right else "only_first",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )


def _cls_index(input_ids: List[int], tokenizer) -> int:
    cls = getattr(tokenizer, "cls_token_id", None)
    if cls is not None and cls in input_ids:
        return input_ids.index(cls)
    return 0


def prepare_train_features(tokenizer, examples: Sequence[Dict[str, Any]], max_length: int = 384,
                           stride: int = 128) -> Dict[str, List[Any]]:
    """Fenster mit start_positions/end_positions (Token-Indizes).

    Liegt die Antwort nicht vollstaendig im Fenster (oder gibt es keine),
    zeigen beide Positionen auf das CLS-Token: "keine Antwort hier".
    """
    enc = _encode(tokenizer, [e["question"] for e in examples], [e["context"] for e in examples],
                  max_length, stride)
    ctx_seq = 1 if _pad_right(tokenizer) else 0
    sample_map = enc.pop("overflow_to_sample_mapping")
    offsets_all = enc.pop("offset_mapping")
    starts, ends = [], []
    for i, offsets in enumerate(offsets_all):
        ids = enc["input_ids"][i]
        cls = _cls_index(ids, tokenizer)
        seq_ids = enc.sequence_ids(i)
        ex = examples[sample_map[i]]
        ans = ex["answers"]
        if not ans["answer_start"]:
            starts.append(cls); ends.append(cls)
            continue
        s_char = ans["answer_start"][0]
        e_char = s_char + len(ans["text"][0])
        tok_s = 0
        while tok_s < len(seq_ids) and seq_ids[tok_s] != ctx_seq:
            tok_s += 1
        tok_e = len(ids) - 1
        while tok_e >= 0 and seq_ids[tok_e] != ctx_seq:
            tok_e -= 1
        if tok_s > tok_e or not (offsets[tok_s][0] <= s_char and offsets[tok_e][1] >= e_char):
            starts.append(cls); ends.append(cls)
            continue
        while tok_s < len(offsets) and seq_ids[tok_s] == ctx_seq and offsets[tok_s][0] <= s_char:
            tok_s += 1
        while tok_e >= 0 and seq_ids[tok_e] == ctx_seq and offsets[tok_e][1] >= e_char:
            tok_e -= 1
        starts.append(tok_s - 1)
        ends.append(tok_e + 1)
    out = {k: list(v) for k, v in enc.items()}
    out["start_positions"] = starts
    out["end_positions"] = ends
    return out


def prepare_eval_features(tokenizer, examples: Sequence[Dict[str, Any]], max_length: int = 384,
                          stride: int = 128) -> Dict[str, List[Any]]:
    """Fenster fuer die Vorhersage: example_id + offset_mapping nur im Kontext."""
    enc = _encode(tokenizer, [e["question"] for e in examples], [e["context"] for e in examples],
                  max_length, stride)
    ctx_seq = 1 if _pad_right(tokenizer) else 0
    sample_map = enc.pop("overflow_to_sample_mapping")
    out = {k: list(v) for k, v in enc.items() if k != "offset_mapping"}
    out["example_id"] = []
    out["offset_mapping"] = []
    for i, offsets in enumerate(enc["offset_mapping"]):
        seq_ids = enc.sequence_ids(i)
        out["example_id"].append(examples[sample_map[i]]["id"])
        out["offset_mapping"].append([tuple(o) if seq_ids[k] == ctx_seq else None
                                      for k, o in enumerate(offsets)])
        out.setdefault("cls_index", []).append(_cls_index(enc["input_ids"][i], tokenizer))
    return out


# ─── Antwort aus den Logits ───────────────────────────────────────────────────

def postprocess(examples: Sequence[Dict[str, Any]], features: Dict[str, List[Any]],
                start_logits, end_logits, n_best: int = 20, max_answer_length: int = 30,
                allow_no_answer: bool = False, null_threshold: float = 0.0
                ) -> Dict[str, Dict[str, Any]]:
    """{id: {"text", "start", "end", "score", "null_score", "candidates"}}.

    Pro Frage gewinnt die beste Spanne ueber alle Fenster (Score = Start- +
    End-Logit). Mit allow_no_answer wird "keine Antwort" gewaehlt, wenn der
    CLS-Score (bestes Fenster) die beste Spanne um mehr als null_threshold
    uebertrifft — wie im SQuAD-v2-Skript von HF.
    """
    import numpy as np

    by_example: Dict[str, List[int]] = collections.defaultdict(list)
    for i, eid in enumerate(features["example_id"]):
        by_example[eid].append(i)

    preds: Dict[str, Dict[str, Any]] = {}
    for ex in examples:
        context = ex["context"]
        cands: List[Dict[str, Any]] = []
        null_score: Optional[float] = None
        for fi in by_example.get(ex["id"], []):
            s_log = np.asarray(start_logits[fi]); e_log = np.asarray(end_logits[fi])
            offsets = features["offset_mapping"][fi]
            cls = features.get("cls_index", [0] * (fi + 1))[fi]
            ns = float(s_log[cls] + e_log[cls])
            null_score = ns if null_score is None else min(null_score, ns)
            s_idx = np.argsort(s_log)[-1:-n_best - 1:-1]
            e_idx = np.argsort(e_log)[-1:-n_best - 1:-1]
            for s in s_idx:
                for e in e_idx:
                    if s >= len(offsets) or e >= len(offsets) or offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_length:
                        continue
                    a, b = offsets[s][0], offsets[e][1]
                    cands.append({"text": context[a:b], "start": int(a), "end": int(b),
                                  "score": float(s_log[s] + e_log[e])})
        cands.sort(key=lambda c: c["score"], reverse=True)
        # gleiche Textstellen aus ueberlappenden Fenstern nur einmal
        uniq, seen = [], set()
        for c in cands:
            key = (c["start"], c["end"])
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        best = uniq[0] if uniq else {"text": "", "start": -1, "end": -1, "score": float("-inf")}
        no_answer = not uniq or (allow_no_answer and null_score is not None
                                 and null_score - best["score"] > null_threshold)
        preds[ex["id"]] = {
            "text": "" if no_answer else best["text"],
            "start": -1 if no_answer else best["start"],
            "end": -1 if no_answer else best["end"],
            "score": best["score"],
            "null_score": null_score if null_score is not None else float("-inf"),
            "candidates": uniq[:5],
        }
    return preds


def span_probability(pred: Dict[str, Any]) -> Optional[float]:
    """Softmax ueber die Top-Kandidaten und 'keine Antwort' -> Anteil der Gewinnerspanne."""
    import math
    scores = [c["score"] for c in pred.get("candidates", [])]
    if pred.get("null_score") not in (None, float("-inf")):
        scores = scores + [pred["null_score"]]
    if not scores:
        return None
    m = max(scores)
    z = sum(math.exp(s - m) for s in scores)
    win = pred["null_score"] if pred["text"] == "" else pred["score"]
    return float(math.exp(win - m) / z)


def mark_answer(context: str, start: int, end: int, width: int = 300) -> str:
    """Kontext mit markierter Antwortstelle, lange Kontexte um die Stelle gekuerzt."""
    if start < 0 or end <= start:
        return context[:width * 2] + ("…" if len(context) > width * 2 else "")
    a = max(0, start - width)
    b = min(len(context), end + width)
    return (("…" if a > 0 else "") + context[a:start] + "»" + context[start:end] + "«"
            + context[end:b] + ("…" if b < len(context) else ""))


# ─── SQuAD-Kennzahlen ─────────────────────────────────────────────────────────

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Wie im offiziellen SQuAD-Skript."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def score_one(pred: str, golds: Sequence[str]) -> Tuple[float, float]:
    """(EM, F1) gegen alle Referenzen; keine Referenz = 'keine Antwort'."""
    golds = [g for g in golds if normalize_answer(g)]
    if not golds:
        ok = float(not normalize_answer(pred))
        return ok, ok
    return max(exact_match(pred, g) for g in golds), max(f1_score(pred, g) for g in golds)


def squad_metrics(preds: Dict[str, str], examples: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """exact_match / f1 (0..1); bei gemischten Daten zusaetzlich HasAns_/NoAns_."""
    em_all, f1_all = [], []
    has, no = [], []
    for ex in examples:
        em, f1 = score_one(preds.get(ex["id"], ""), ex["answers"]["text"])
        em_all.append(em); f1_all.append(f1)
        (has if ex["answers"]["text"] else no).append((em, f1))
    if not em_all:
        return {}
    out = {"exact_match": sum(em_all) / len(em_all), "f1": sum(f1_all) / len(f1_all)}
    if has and no:
        out["has_answer_exact_match"] = sum(e for e, _ in has) / len(has)
        out["has_answer_f1"] = sum(f for _, f in has) / len(has)
        out["no_answer_accuracy"] = sum(e for e, _ in no) / len(no)
    return out


# ─── Spec neben dem Modell ────────────────────────────────────────────────────

def save_spec(model_dir: Path, spec: Dict[str, Any]) -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / SPEC_FILE).write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")


def load_spec(model_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((Path(model_dir) / SPEC_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def parse_single_input(text: str, question: str = "") -> Tuple[str, str]:
    """Einzeltest -> (context, question).

    Formen: Kontext im Textfeld + Frage im zweiten Feld; "Frage ||| Kontext";
    JSON {"question": ..., "context": ...}.
    """
    raw = (text or "").strip()
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("context") and obj.get("question"):
                return str(obj["context"]), str(obj["question"])
        except ValueError:
            pass
    if question.strip():
        return raw, question.strip()
    if "|||" in raw:
        q, c = (p.strip() for p in raw.split("|||", 1))
        return c, q
    raise ValueError('Frage fehlt. Kontext eingeben und die Frage ins zweite Feld schreiben '
                     '(oder "Frage ||| Kontext").')
