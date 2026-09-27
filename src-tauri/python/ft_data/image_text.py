"""Bild-Text-Datasets: Bild + Beschreibung bzw. Bild + Frage + Antwort.

Genutzt von zwei Plugins, in Training UND Test, damit beide dieselben Zeilen
sehen:
  * text_to_image_lora  – Bild + Caption (Diffusion-LoRA, "eigener Stil")
  * vision_language     – Bild + Frage + Antwort (VLM-Feintuning)

Unterstuetzte Layouts (je Split-Ordner train/ val/ test/ oder direkt im Root):
    metadata.jsonl | metadata.csv | *.jsonl | *.json | *.csv | *.tsv
        {"file_name": "a.png", "text": "ein roter Kreis"}                (Caption)
        {"image": "img/a.png", "question": "Welche Farbe?", "answer": "rot"}
        {"image": "a.png", "messages": [{"role": "user", "content": [{"type": "image"},
            {"type": "text", "text": "Welche Farbe?"}]}, {"role": "assistant", "content": "rot"}]}
    a.png + a.txt                               (Caption im gleichnamigen .txt)
    <klasse>/a.png                              (nur VLM: Antwort = Ordnername)
    nur Bilder                                  (nur Diffusion: instance_prompt fuer alle)
    *.parquet mit Bildbytes + Textspalten       (HF-Download; wird einmalig entpackt)

Bildpfade in Tabellen sind relativ zur Tabelle; ein Pfad relativ zum Dataset-
Root wird ebenfalls gefunden, weil HF-Exporte beides machen.
"""
from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .media import (
    IMAGE_EXTS, MEDIA_CACHE_DIR, SPLIT_ALIASES, _guess_ext, _parquet_files,
    _signature, _visible_dirs,
)

TABLE_EXTS = (".jsonl", ".json", ".csv", ".tsv")

# Reihenfolge = Vorrang. "file_name" ist die HF-imagefolder-Konvention.
IMAGE_KEYS = ("file_name", "image", "image_path", "img", "filename", "file", "path", "image_file")
QUESTION_KEYS = ("question", "prompt", "query", "instruction", "input")
# VLM-Antwort. "text" steht hinten: bei Caption-Datasets ist es die Antwort,
# bei Frage/Antwort-Zeilen aber oft die Frage selbst.
ANSWER_KEYS = ("answer", "answers", "caption", "response", "output", "target", "label", "text")
# Diffusion: die Beschreibung des Bildes. Hier ist "prompt" die Caption.
CAPTION_KEYS = ("text", "caption", "prompt", "captions", "description", "answer")

# Dateien, die die Plugins selbst schreiben — kein Trainingsmaterial.
_SKIP_DIRS = {MEDIA_CACHE_DIR, "samples", "__pycache__"}


@dataclass
class ImageTextSample:
    image: Path
    prompt: str = ""   # Frage / Anweisung (VLM). Leer = Standardprompt des Plugins.
    answer: str = ""   # Antwort (VLM) bzw. Caption (Diffusion)

    def as_dict(self) -> Dict[str, str]:
        return {"image": str(self.image), "prompt": self.prompt, "answer": self.answer}


@dataclass
class ImageTextSplits:
    train: List[ImageTextSample]
    val: List[ImageTextSample]
    test: List[ImageTextSample]
    source: str
    notes: List[str] = field(default_factory=list)


# ── Zeilen lesen ─────────────────────────────────────────────────────────────

def read_table(path: Path) -> List[Dict[str, Any]]:
    """JSONL, JSON (Liste oder {"data": [...]}), CSV oder TSV als Liste von Zeilen."""
    path = Path(path)
    ext = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if ext == ".jsonl":
        rows = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path.name}, Zeile {i}: kein gueltiges JSON ({exc})") from exc
            if isinstance(obj, dict):
                rows.append(obj)
        return rows
    if ext == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("data", "rows", "items", "annotations", "samples"):
                if isinstance(obj.get(key), list):
                    obj = obj[key]
                    break
        return [r for r in obj if isinstance(r, dict)] if isinstance(obj, list) else []
    delim = "\t" if ext == ".tsv" else ","
    return [dict(r) for r in csv.DictReader(text.splitlines(), delimiter=delim)]


def _first(row: Dict[str, Any], keys: Sequence[str]) -> Tuple[Optional[str], Any]:
    lowered = {str(k).lower(): k for k in row}
    for key in keys:
        real = lowered.get(key)
        if real is not None and row[real] not in (None, "", []):
            return real, row[real]
    return None, None


def _as_text(value: Any) -> str:
    """Antworten kommen auch als Liste (VQA: mehrere Annotatoren) — dann die erste."""
    if isinstance(value, list):
        value = next((v for v in value if v not in (None, "")), "")
        if isinstance(value, dict):
            value = value.get("answer") or value.get("text") or ""
    return str(value).strip() if value is not None else ""


def _content_text(content: Any) -> Tuple[str, Optional[str]]:
    """Text und ggf. Bildpfad aus einem Chat-Inhalt (String oder Liste von Teilen)."""
    if isinstance(content, str):
        return content.replace("<image>", "").strip(), None
    text_parts: List[str] = []
    image: Optional[str] = None
    for part in content or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text" and part.get("text"):
            text_parts.append(str(part["text"]))
        elif kind == "image":
            for key in ("image", "path", "url", "image_url"):
                val = part.get(key)
                if isinstance(val, str) and val:
                    image = val
                    break
    return " ".join(t.strip() for t in text_parts if t.strip()), image


def parse_messages(messages: Any) -> Tuple[str, str, Optional[str]]:
    """(Frage, Antwort, Bildpfad) aus dem Chat-Format. Der erste Nutzer-Zug ist
    die Frage, der erste Assistenten-Zug danach die Antwort."""
    question, answer, image = "", "", None
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower()
        text, img = _content_text(msg.get("content"))
        image = image or img
        if role == "user" and not question:
            question = text
        elif role == "assistant" and question and not answer:
            answer = text
    return question, answer, image


def _resolve_image(raw: Any, bases: Sequence[Path]) -> Optional[Path]:
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, dict):
        raw = raw.get("path") or raw.get("file_name")
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw.strip())
    if p.is_absolute():
        return p if p.exists() else None
    for base in bases:
        cand = base / p
        if cand.exists():
            return cand
    # Manche Exporte speichern "images/a.png", legen die Bilder aber flach ab.
    for base in bases:
        cand = base / p.name
        if cand.exists():
            return cand
    return None


def row_to_sample(row: Dict[str, Any], bases: Sequence[Path], mode: str) -> Optional[ImageTextSample]:
    """Eine Tabellenzeile als Sample; None, wenn Bild oder Text fehlen.

    mode "vlm": Frage (optional) + Antwort. mode "caption": nur die Beschreibung.
    """
    image_raw: Any = None
    question, answer = "", ""
    _, messages = _first(row, ("messages", "conversations", "conversation"))
    if isinstance(messages, list):
        question, answer, image_raw = parse_messages(_normalise_sharegpt(messages))
    if image_raw is None:
        _, image_raw = _first(row, IMAGE_KEYS + ("images",))
    image = _resolve_image(image_raw, bases)
    if image is None:
        return None
    if mode == "caption":
        if not answer:
            _, cap = _first(row, CAPTION_KEYS)
            answer = _as_text(cap)
        return ImageTextSample(image, "", answer) if answer else None
    if not question:
        q_key, q = _first(row, QUESTION_KEYS)
        question = _as_text(q)
    else:
        q_key = None
    if not answer:
        # Die Frage-Spalte darf nicht zugleich die Antwort sein.
        keys = tuple(k for k in ANSWER_KEYS if k != (q_key or "").lower())
        _, a = _first(row, keys)
        answer = _as_text(a)
    return ImageTextSample(image, question, answer) if answer else None


def _normalise_sharegpt(messages: List[Any]) -> List[Any]:
    """ShareGPT/LLaVA ({"from": "human", "value": ...}) auf role/content abbilden."""
    out = []
    for m in messages:
        if isinstance(m, dict) and "from" in m and "role" not in m:
            role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}.get(
                str(m.get("from")).lower(), str(m.get("from")))
            out.append({"role": role, "content": m.get("value", "")})
        else:
            out.append(m)
    return out


# ── Ordner lesen ─────────────────────────────────────────────────────────────

def _images_in(d: Path) -> List[Path]:
    out = []
    for f in sorted(d.rglob("*")):
        if not f.is_file() or f.name.startswith(".") or f.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = f.relative_to(d).parts
        if any(p in _SKIP_DIRS or p.startswith(".") for p in rel[:-1]):
            continue
        out.append(f)
    return out


def _tables_in(d: Path) -> List[Path]:
    """Tabellen direkt im Ordner; metadata.* zuerst (HF-imagefolder-Konvention)."""
    files = [f for f in sorted(d.iterdir()) if f.is_file() and f.suffix.lower() in TABLE_EXTS
             and not f.name.startswith(".")] if d.is_dir() else []
    # Eigene Beipackzettel und Konfigurationen sind keine Daten.
    files = [f for f in files if f.name.lower() not in {
        "provenance.csv", "label_mapping.json", "dataset_info.json", "config.json",
        "dataset.json", "state.json"}]
    files.sort(key=lambda f: (0 if f.stem.lower() == "metadata" else 1, f.name))
    return files


def samples_from_dir(d: Path, mode: str, instance_prompt: str = "",
                     tables: Optional[List[Path]] = None,
                     notes: Optional[List[str]] = None) -> List[ImageTextSample]:
    """Alle Samples eines Ordners (ein Split oder das ganze Dataset)."""
    notes = notes if notes is not None else []
    tables = _tables_in(d) if tables is None else tables
    samples: List[ImageTextSample] = []
    if tables:
        skipped = 0
        for t in tables:
            try:
                rows = read_table(t)
            except (ValueError, OSError, UnicodeDecodeError) as exc:
                notes.append(f"{t.name} uebersprungen: {exc}")
                continue
            for row in rows:
                s = row_to_sample(row, (t.parent, d), mode)
                if s is None:
                    skipped += 1
                else:
                    samples.append(s)
        if samples:
            if skipped:
                notes.append(f"{skipped} Zeilen ohne auffindbares Bild oder ohne Text uebersprungen.")
            return samples
        notes.append("Tabellen ohne verwertbare Zeilen (Bild + Text) — lese Bilder und .txt-Dateien.")

    missing = 0
    for img in _images_in(d):
        txt = img.with_suffix(".txt")
        text = txt.read_text(encoding="utf-8", errors="replace").strip() if txt.exists() else ""
        rel = img.relative_to(d).parts
        if not text and mode == "vlm" and len(rel) >= 2:
            # Klassenordner: die Antwort ist der Ordnername.
            text = rel[0].replace("_", " ")
        if not text:
            text = instance_prompt.strip()
        if not text:
            missing += 1
            continue
        samples.append(ImageTextSample(img, "", text))
    if missing:
        hint = ("Lege gleichnamige .txt-Dateien an oder setze einen instance_prompt."
                if mode == "caption" else "Lege gleichnamige .txt-Dateien oder eine metadata.jsonl an.")
        notes.append(f"{missing} Bilder ohne Beschreibung uebersprungen. {hint}")
    return samples


def _split_of(name: str) -> Optional[str]:
    low = name.lower()
    for canon, aliases in SPLIT_ALIASES.items():
        if any(re.search(rf"(^|[^a-z]){a}([^a-z]|$)", low) for a in aliases):
            return canon
    return None


def resolve_image_text(root: Path, mode: str, instance_prompt: str = "", seed: int = 42,
                       val_fraction: float = 0.1, status: Optional[Callable[[str], None]] = None
                       ) -> ImageTextSplits:
    """Liest ein Bild-Text-Dataset mit Splits.

    val_fraction > 0: fehlt ein val-Split, wird dieser Anteil aus train
    abgetrennt (test/ bleibt unberuehrt). Diffusion setzt 0 — bei zehn
    DreamBooth-Bildern waere jedes abgetrennte Bild ein Verlust.
    """
    root = ensure_image_text_folders(Path(root), status)
    if not root.exists():
        raise FileNotFoundError(f"Dataset-Pfad existiert nicht: {root}")
    notes: List[str] = []
    split: Dict[str, List[ImageTextSample]] = {}

    # 1. Split-Ordner (train/ val/ test/)
    by_name = {d.name.lower(): d for d in _visible_dirs(root)}
    for canon, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            d = by_name.get(alias)
            if d is not None:
                got = samples_from_dir(d, mode, instance_prompt, notes=notes)
                if got:
                    split[canon] = got
                    break

    # 2. Tabellen im Root, nach Namen aufgeteilt (train.jsonl, validation.csv ...)
    if not split:
        tables = _tables_in(root)
        named = {t: _split_of(t.stem) for t in tables}
        if tables and any(v for v in named.values()):
            for t, canon in named.items():
                got = samples_from_dir(root, mode, instance_prompt, tables=[t], notes=notes)
                if got:
                    split.setdefault(canon or "train", []).extend(got)

    source = "Split-Ordner" if split else ""
    if not split:
        got = samples_from_dir(root, mode, instance_prompt, notes=notes)
        if got:
            split["train"] = got
        source = "ungeteilt"

    if not split.get("train"):
        found = sum(len(v) for v in split.values())
        if found:
            raise ValueError(
                f"In '{root}' gibt es Daten ({', '.join(sorted(split))}), aber keinen train-Split.")
        hint = ("Erwartet: Bilder mit gleichnamigen .txt-Captions, eine metadata.jsonl/.csv "
                "(file_name + text) oder nur Bilder plus instance_prompt."
                if mode == "caption" else
                "Erwartet: JSONL mit image + question + answer (oder messages), Bilder mit "
                "gleichnamigen .txt-Dateien oder ein Ordner pro Antwort.")
        detail = f" Hinweise: {' '.join(notes)}" if notes else ""
        raise ValueError(f"Keine Bild-Text-Paare in '{root}' gefunden. {hint}{detail}")

    train = split["train"]
    val = split.get("val", [])
    test = split.get("test", [])
    if not val and val_fraction > 0 and len(train) > 1:
        rng = random.Random(seed)
        shuffled = train[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        val, train = shuffled[:n_val], shuffled[n_val:]
        notes.append(f"Kein val-Split — {len(val)} Beispiele ({val_fraction:.0%}) aus train abgetrennt.")
    return ImageTextSplits(train, val, test, source, notes)


def evaluation_samples(root: Path, mode: str, instance_prompt: str = "",
                       status: Optional[Callable[[str], None]] = None) -> Tuple[List[ImageTextSample], str]:
    """Samples fuer einen Testlauf: test, sonst val, sonst alles."""
    splits = resolve_image_text(root, mode, instance_prompt, val_fraction=0.0, status=status)
    if splits.test:
        return splits.test, "test"
    if splits.val:
        return splits.val, "val"
    return splits.train, "train"


# ── HF-Parquet mit Bildbytes + Text ─────────────────────────────────────────

def _image_column(schema) -> Optional[str]:
    import pyarrow as pa
    from .media import _hf_features

    feats = _hf_features(schema)
    col = next((n for n, f in feats.items() if isinstance(f, dict) and f.get("_type") == "Image"), None)
    if col:
        return col
    for f in schema:
        if f.name.lower() in ("image", "img", "picture", "images") and (
                pa.types.is_struct(f.type) or pa.types.is_binary(f.type) or pa.types.is_list(f.type)):
            return f.name
    return None


def ensure_image_text_folders(root: Path, status: Optional[Callable[[str], None]] = None) -> Path:
    """Entpackt HF-Parquet mit Bildern einmalig nach <root>/.frametrain_media/imagetext/.

    Je Split entsteht ein Ordner mit den Bildern und einer metadata.jsonl, die
    alle Textspalten der Parquet-Zeile enthaelt — die Spaltenwahl bleibt damit
    bei row_to_sample und ist dieselbe wie fuer JSONL-Datasets.
    """
    root = Path(root)
    if not root.is_dir() or _images_in(root):
        return root
    parquet = _parquet_files(root)
    if not parquet:
        return root
    import pyarrow.parquet as pq

    first = next(iter(parquet.values()))[0]
    schema = pq.read_schema(first)
    img_col = _image_column(schema)
    if img_col is None:
        return root
    cache = root / MEDIA_CACHE_DIR / "imagetext"
    marker = cache / ".complete.json"
    sig = _signature(parquet)
    try:
        if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("signature") == sig:
            return cache
    except ValueError:
        pass
    import shutil
    if cache.exists():
        shutil.rmtree(cache)
    if status:
        status("Entpacke Bilder und Texte aus Parquet (einmalig)...")
    counts: Dict[str, int] = {}
    for split_name, files in parquet.items():
        target = cache / split_name
        target.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(target / "metadata.jsonl", "w", encoding="utf-8") as meta:
            for f in files:
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=128):
                    for row in batch.to_pylist():
                        value = row.pop(img_col, None)
                        if isinstance(value, list):
                            value = value[0] if value else None
                        data, hint = (value.get("bytes"), value.get("path")) if isinstance(value, dict) else (value, None)
                        if not data:
                            continue
                        name = f"{n:07d}{_guess_ext(data, hint, 'image')}"
                        (target / name).write_bytes(data)
                        clean = {k: v for k, v in row.items() if _jsonable(v)}
                        clean["file_name"] = name
                        meta.write(json.dumps(clean, ensure_ascii=False) + "\n")
                        n += 1
        counts[split_name] = n
    marker.write_text(json.dumps({"signature": sig, "counts": counts}), encoding="utf-8")
    if status:
        status("Parquet entpackt: " + ", ".join(f"{s}: {c}" for s, c in counts.items()))
    return cache


def _jsonable(v: Any) -> bool:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return True
    if isinstance(v, list):
        return all(_jsonable(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _jsonable(x) for k, x in v.items())
    return False


# ── Kennzahlen fuer generierten Text ────────────────────────────────────────

def normalise_answer(text: str) -> str:
    """Vergleichsform: klein, ohne Satzzeichen und doppelte Leerzeichen."""
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def text_scores(predictions: Iterable[str], references: Iterable[str]) -> Dict[str, float]:
    """exact_match (nach normalise_answer) und ROUGE-L-F1 im Mittel.

    ROUGE-L misst die laengste gemeinsame Teilfolge — fuer Captions und freie
    Antworten aussagekraeftiger als der reine Wortgleich-Vergleich.
    """
    preds, refs = list(predictions), list(references)
    if not preds or len(preds) != len(refs):
        return {}
    exact = sum(1 for p, r in zip(preds, refs) if normalise_answer(p) == normalise_answer(r))
    out = {"exact_match": exact / len(preds)}
    f1 = [rouge_l_f1(normalise_answer(p), normalise_answer(r)) for p, r in zip(preds, refs)]
    out["rougeL"] = sum(f1) / len(f1)
    return out


def rouge_l_f1(prediction: str, reference: str) -> float:
    """ROUGE-L-F1 auf Wortebene (laengste gemeinsame Teilfolge).

    Selbst gerechnet statt ueber rouge_score: das Paket ist nicht ueberall
    installiert, und sein Tokenizer kennt nur [a-z0-9] — "grün" zerfiel in
    "gr" und "n". Rechnet wie rouge_score ohne Stemmer (beta = 1).
    """
    p, r = prediction.split(), reference.split()
    if not p or not r:
        return 1.0 if not p and not r else 0.0
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r, start=1):
            cur.append(prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)
