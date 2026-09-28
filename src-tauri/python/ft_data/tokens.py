"""Token-Klassifikation (NER, POS): Daten lesen, Labels ausrichten, Entitaeten bilden.

Gemeinsam fuer Training, Test und Modell-Server — damit Wortgrenzen,
Label-Reihenfolge und das Zusammenfassen zu Entitaeten ueberall gleich sind.

Unterstuetzte Formate:
  * JSONL/JSON/Parquet mit einer Token-Liste (tokens/words) und einer
    Tag-Liste (ner_tags/labels/tags/pos_tags). Tags als Strings oder ints;
    bei ints liefern HF-Parquet-Dateien die Namen als ClassLabel mit.
  * CoNLL-Text (.conll, .txt, .iob, .bio): ein Token pro Zeile, Tag in der
    letzten Spalte, Leerzeile = Satzgrenze, -DOCSTART- wird uebersprungen.
  * Spans: {"text": "...", "entities": [{"start", "end", "label"}]} — wird in
    Woerter mit BIO-Tags umgewandelt.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TABLE_EXTS = (".jsonl", ".json", ".parquet")
CONLL_EXTS = (".conll", ".txt", ".iob", ".bio", ".conllu")
DATA_EXTS = TABLE_EXTS + CONLL_EXTS

TOKEN_COLUMNS = ["tokens", "words", "token", "sentence_tokens"]
TAG_COLUMNS = ["ner_tags", "labels", "tags", "ner", "pos_tags", "upos", "label", "tag"]
SPEC_FILE = "frametrain_token.json"

# Woerter und einzelne Satzzeichen — "Berlin," sind zwei Tokens. Bindestrich-
# und Apostroph-Woerter bleiben zusammen ("Baden-Wuerttemberg", "don't").
WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)

BIO_PREFIXES = ("B-", "I-", "E-", "S-", "L-", "U-")


# ── Lesen ───────────────────────────────────────────────────────────────────

def split_words(text: str) -> List[Tuple[str, int, int]]:
    """Text -> [(wort, start, ende)] mit Zeichenpositionen."""
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text or "")]


def spans_to_bio(text: str, entities: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Text + Zeichen-Spans -> Woerter + BIO-Tags.

    Ein Wort gehoert zu einer Entitaet, sobald es sich mit ihr ueberschneidet —
    Spans, die mitten im Wort enden (Tippfehler beim Annotieren), gehen so
    nicht verloren.
    """
    words = split_words(text)
    tags = ["O"] * len(words)
    ents = sorted(
        (e for e in entities or [] if isinstance(e, dict)),
        key=lambda e: int(e.get("start", 0)),
    )
    for ent in ents:
        try:
            s, e = int(ent["start"]), int(ent["end"])
        except (KeyError, TypeError, ValueError):
            continue
        label = str(ent.get("label") or ent.get("type") or ent.get("entity") or "ENT")
        first = True
        for i, (_, ws, we) in enumerate(words):
            if we <= s or ws >= e or tags[i] != "O":
                continue
            tags[i] = ("B-" if first else "I-") + label
            first = False
    return [w for w, _, _ in words], tags


def read_conll(path: Path) -> List[Dict[str, List[str]]]:
    """CoNLL-Datei -> [{"tokens": [...], "tags": [...]}]."""
    rows: List[Dict[str, List[str]]] = []
    toks: List[str] = []
    tags: List[str] = []

    def flush():
        if toks:
            rows.append({"tokens": list(toks), "tags": list(tags)})
        toks.clear()
        tags.clear()

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                flush()
                continue
            if line.startswith("-DOCSTART-") or line.startswith("#"):
                # CoNLL-U-Kommentare und Dokumentgrenzen sind keine Tokens.
                if line.startswith("-DOCSTART-"):
                    flush()
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                # Nur ein Token ohne Tag: als "O" werten statt den Satz zu verwerfen.
                toks.append(parts[0])
                tags.append("O")
                continue
            if path.suffix.lower() == ".conllu" and len(parts) >= 4:
                # CoNLL-U: FORM in Spalte 2, UPOS in Spalte 4. Mehrwort-Zeilen (1-2) ueberspringen.
                if "-" in parts[0] or "." in parts[0]:
                    continue
                toks.append(parts[1])
                tags.append(parts[3])
                continue
            toks.append(parts[0])
            tags.append(parts[-1])
    flush()
    return rows


def _label_names_from_features(features: Any, column: str) -> Optional[List[str]]:
    """ClassLabel-Namen einer Listen-Spalte (HF-Datasets), falls vorhanden."""
    try:
        feat = features[column]
    except Exception:
        return None
    inner = getattr(feat, "feature", None)
    names = getattr(inner, "names", None) or getattr(feat, "names", None)
    return list(names) if names else None


def read_table(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """JSON/JSONL/Parquet -> (Zeilen, {spalte: ClassLabel-Namen})."""
    ext = path.suffix.lower()
    if ext == ".parquet":
        from datasets import Dataset
        ds = Dataset.from_parquet(str(path))
        names = {}
        for col in ds.column_names:
            n = _label_names_from_features(ds.features, col)
            if n:
                names[col] = n
        return [dict(r) for r in ds], names
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    stripped = text.lstrip()
    if ext == ".json" and stripped.startswith(("[", "{")):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # {"data": [...]} oder {"train": [...]} — die erste Liste nehmen.
                data = next((v for v in data.values() if isinstance(v, list)), [data])
            return [r for r in data if isinstance(r, dict)], {}
        except ValueError:
            pass  # Doch JSON Lines mit .json-Endung
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows, {}


def resolve_columns(columns: Sequence[str], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
    """Welche Spalten tragen Tokens und Tags? Oder ist es das Spans-Format?"""
    cfg = cfg or {}
    cols = list(columns)
    tok = cfg.get("tokens_column") or next((c for c in TOKEN_COLUMNS if c in cols), None)
    tag = cfg.get("tags_column") or next((c for c in TAG_COLUMNS if c in cols and c != tok), None)
    if tok and tag and tok in cols and tag in cols:
        return {"format": "tokens", "tokens_column": tok, "tags_column": tag}
    text_col = cfg.get("text_column") or next((c for c in ("text", "sentence", "content") if c in cols), None)
    ent_col = cfg.get("entities_column") or next(
        (c for c in ("entities", "spans", "annotations", "label", "labels") if c in cols), None)
    if text_col and ent_col:
        return {"format": "spans", "text_column": text_col, "entities_column": ent_col}
    raise ValueError(
        f"Token- und Tag-Spalte nicht erkannt. Vorhandene Spalten: {cols}.\n"
        "Erwartet: tokens + ner_tags (bzw. labels/tags) als Listen, oder text + entities "
        "mit {start, end, label}. Eigene Namen in der Plugin-Konfiguration setzen, z.B.:\n"
        '  {"tokens_column": "woerter", "tags_column": "etiketten"}'
    )


def _tag_name(value: Any, names: Optional[List[str]], cfg_names: Optional[List[str]]) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) or (isinstance(value, float) and value == int(value)):
        idx = int(value)
        for table in (names, cfg_names):
            if table and 0 <= idx < len(table):
                return str(table[idx])
        return str(idx)
    return str(value)


def normalize_rows(rows: Sequence[Dict[str, Any]], spec: Dict[str, Any],
                   class_names: Optional[Dict[str, List[str]]] = None,
                   cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, List[str]]]:
    """Zeilen beliebigen Formats -> [{"tokens": [...], "tags": [...]}] mit String-Tags."""
    cfg = cfg or {}
    class_names = class_names or {}
    cfg_names = cfg.get("label_list") if isinstance(cfg.get("label_list"), list) else None
    out: List[Dict[str, List[str]]] = []
    if spec.get("format") == "spans":
        for r in rows:
            text = r.get(spec["text_column"])
            ents = r.get(spec["entities_column"])
            if isinstance(ents, str):
                try:
                    ents = json.loads(ents)
                except ValueError:
                    ents = []
            if not isinstance(text, str) or not text.strip():
                continue
            toks, tags = spans_to_bio(text, ents if isinstance(ents, list) else [])
            if toks:
                out.append({"tokens": toks, "tags": tags})
        return out
    names = class_names.get(spec["tags_column"])
    for r in rows:
        toks = r.get(spec["tokens_column"])
        tags = r.get(spec["tags_column"])
        if isinstance(toks, str):
            toks = toks.split()
        if isinstance(tags, str):
            tags = tags.split()
        if not toks or tags is None:
            continue
        toks, tags = list(toks), list(tags)
        if len(toks) != len(tags):
            # Kaputte Zeile lieber auslassen als Labels um eine Stelle verschoben lernen.
            continue
        out.append({"tokens": [str(t) for t in toks],
                    "tags": [_tag_name(t, names, cfg_names) for t in tags]})
    return out


def load_files(files: Iterable[Path], cfg: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, List[str]]], Dict[str, Any]]:
    """Liest alle Dateien eines Splits. -> (Saetze, erkannte Spalten)"""
    cfg = cfg or {}
    sentences: List[Dict[str, List[str]]] = []
    spec: Dict[str, Any] = {}
    for f in files:
        f = Path(f)
        if f.suffix.lower() in CONLL_EXTS:
            got = read_conll(f)
            if got and not spec:
                spec = {"format": "conll"}
            sentences.extend(got)
            continue
        rows, names = read_table(f)
        if not rows:
            continue
        file_spec = resolve_columns(list(rows[0].keys()), cfg)
        if not spec or spec.get("format") == "conll":
            spec = file_spec
        sentences.extend(normalize_rows(rows, file_spec, names, cfg))
    return sentences, spec


# ── Labels ──────────────────────────────────────────────────────────────────

def is_bio_scheme(labels: Iterable[str]) -> bool:
    """True, wenn alle Tags ausser O ein BIO/IOBES-Praefix tragen (NER).

    POS-Tags (NOUN, VERB) haben keins — dort misst seqeval nichts Sinnvolles,
    es wird dann pro Token gezaehlt.
    """
    rest = [l for l in labels if l != "O"]
    return bool(rest) and all(l.startswith(BIO_PREFIXES) for l in rest)


def build_label_list(sentences: Iterable[Dict[str, List[str]]]) -> List[str]:
    """Alle Tags, "O" zuerst, danach nach Entitaetstyp sortiert (B- vor I-).

    Fehlt zu einem I-X das B-X (kommt in Spans-Daten mit Ein-Wort-Entitaeten
    nie vor, in CoNLL mit IOB1 schon), wird es ergaenzt — sonst kann das
    Modell spaeter kein korrektes BIO erzeugen.
    """
    seen = set()
    for s in sentences:
        seen.update(s["tags"])
    if is_bio_scheme(seen):
        for l in list(seen):
            if l.startswith("I-"):
                seen.add("B-" + l[2:])

    def key(l: str):
        if l == "O":
            return (0, "", "")
        if l[:2] in BIO_PREFIXES:
            return (1, l[2:], l[:2])
        return (1, l, "")
    return sorted(seen, key=key)


def align_labels(word_ids: Sequence[Optional[int]], word_label_ids: Sequence[int],
                 label_all_tokens: bool = False, b_to_i: Optional[Dict[int, int]] = None) -> List[int]:
    """Wort-Labels auf Subword-Tokens verteilen.

    Nur das erste Subtoken eines Wortes traegt das Label, der Rest -100 (wird
    im Loss ignoriert). Sonst zaehlte "Karolina" als drei Entitaeten, und die
    Metrik rechnete mit Subwords statt Woertern.
    """
    out: List[int] = []
    prev = None
    for wid in word_ids:
        if wid is None:
            out.append(-100)
        elif wid != prev:
            out.append(int(word_label_ids[wid]) if wid < len(word_label_ids) else -100)
        else:
            if label_all_tokens and wid < len(word_label_ids):
                lab = int(word_label_ids[wid])
                out.append(b_to_i.get(lab, lab) if b_to_i else lab)
            else:
                out.append(-100)
        prev = wid
    return out


# ── Metriken ────────────────────────────────────────────────────────────────

def score_sequences(true_tags: List[List[str]], pred_tags: List[List[str]]) -> Dict[str, float]:
    """precision/recall/f1 (Entitaeten bei BIO, sonst pro Token) + Token-Accuracy."""
    flat_t = [t for s in true_tags for t in s]
    flat_p = [p for s in pred_tags for p in s]
    acc = (sum(1 for a, b in zip(flat_t, flat_p) if a == b) / len(flat_t)) if flat_t else 0.0
    labels = set(flat_t) | set(flat_p)
    if is_bio_scheme(set(flat_t)):
        try:
            from seqeval.metrics import f1_score, precision_score, recall_score
        except ImportError:
            from ft_data.deps import missing
            raise missing("seqeval", what="Die NER-Auswertung")
        # Vorhergesagte Tags ohne Praefix (kommt bei einem frischen Kopf vor)
        # wuerde seqeval mit einer Warnung ueberspringen — als O werten.
        pred_clean = [[p if (p == "O" or p.startswith(BIO_PREFIXES)) else "O" for p in s] for s in pred_tags]
        return {
            "precision": float(precision_score(true_tags, pred_clean, zero_division=0)),
            "recall": float(recall_score(true_tags, pred_clean, zero_division=0)),
            "f1": float(f1_score(true_tags, pred_clean, zero_division=0)),
            "accuracy": float(acc),
            "scheme": "entity",
        }
    # POS & Co.: gewichtete Werte pro Token, "O" (falls vorhanden) ausgenommen.
    try:
        from sklearn.metrics import precision_recall_fscore_support
        keep = sorted(l for l in labels if l != "O") or sorted(labels)
        p, r, f, _ = precision_recall_fscore_support(flat_t, flat_p, labels=keep,
                                                     average="weighted", zero_division=0)
    except ImportError:
        p = r = f = acc
    return {"precision": float(p), "recall": float(r), "f1": float(f),
            "accuracy": float(acc), "scheme": "token"}


# ── Inferenz ────────────────────────────────────────────────────────────────

def predict_words(model, tokenizer, words: List[str], id2label: Dict[int, str],
                  device=None, max_length: int = 512) -> List[Tuple[str, float]]:
    """Woerter -> [(tag, wahrscheinlichkeit)] je Wort (erstes Subtoken zaehlt).

    Zu lange Saetze werden abgeschnitten; die Woerter dahinter bekommen "O"
    mit Score 0 — sichtbar statt stillschweigend verschluckt.
    """
    import torch

    if not words:
        return []
    enc = tokenizer(words, is_split_into_words=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    word_ids = enc.word_ids(0)
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits.float(), dim=-1).cpu()
    result: List[Tuple[str, float]] = [("O", 0.0)] * len(words)
    seen = set()
    for pos, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        score, idx = torch.max(probs[pos], dim=-1)
        result[wid] = (str(id2label.get(int(idx), str(int(idx)))), float(score))
    return result


def entities_from_tags(words: List[Tuple[str, int, int]] | List[str], tags: List[str],
                       scores: Optional[List[float]] = None, text: Optional[str] = None,
                       bio: Optional[bool] = None) -> List[Dict[str, Any]]:
    """BIO-Tags je Wort -> zusammenhaengende Entitaeten.

    words: entweder [(wort, start, ende)] (dann stimmen die Zeichenpositionen
    und der Entitaetstext kommt aus `text`) oder reine Woerter.
    Ohne BIO-Schema (POS) wird jedes Wort mit Tag != O einzeln gemeldet.
    bio: Schema aus der Label-Liste des Modells; None = aus den Tags raten.
    """
    scores = scores or [1.0] * len(tags)
    with_pos = bool(words) and isinstance(words[0], tuple)
    if bio is None:
        bio = is_bio_scheme(set(tags)) or all(t == "O" for t in tags)
    ents: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    def close():
        nonlocal cur
        if cur:
            cur["score"] = float(sum(cur["_s"]) / len(cur["_s"]))
            del cur["_s"]
            if with_pos and text is not None:
                cur["text"] = text[cur["start"]:cur["end"]]
            else:
                cur["text"] = " ".join(cur["_w"])
            del cur["_w"]
            ents.append(cur)
        cur = None

    for i, tag in enumerate(tags):
        w = words[i]
        word, ws, we = (w if with_pos else (w, i, i + 1))
        if tag == "O":
            close()
            continue
        if not bio:
            close()
            cur = {"label": tag, "start": ws, "end": we, "_s": [scores[i]], "_w": [word]}
            close()
            continue
        prefix, typ = tag[:2], tag[2:]
        starts_new = prefix in ("B-", "S-", "U-") or cur is None or cur["label"] != typ
        if starts_new:
            close()
            cur = {"label": typ, "start": ws, "end": we, "_s": [scores[i]], "_w": [word]}
        else:
            cur["end"] = we
            cur["_s"].append(scores[i])
            cur["_w"].append(word)
        if prefix in ("E-", "L-", "S-", "U-"):
            close()
    close()
    return ents


def format_entities(ents: List[Dict[str, Any]]) -> str:
    """"Karol [PER], Berlin [LOC]" — so steht das Ergebnis im Test und im Labor."""
    return ", ".join(f"{e['text']} [{e['label']}]" for e in ents)


# ── Spec neben dem Modell ───────────────────────────────────────────────────

def save_spec(model_dir: Path, spec: Dict[str, Any]) -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / SPEC_FILE).write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")


def load_spec(model_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((Path(model_dir) / SPEC_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
