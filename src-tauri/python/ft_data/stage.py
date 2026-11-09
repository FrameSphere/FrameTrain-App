"""Ein Dataset fuer den Dataset Builder (Studio) vorbereiten – fuer alle Datenarten.

Aufruf:  python -m ft_data.stage '{"source": "<dataset-ordner>", "project_kind": "image_boxes",
          "target": "<leerer ordner>", "class_map": {"truck": "vehicle", "bus": "vehicle",
          "car": "vehicle", "person": "person"}, "keep_unmapped": false, "max_samples": 300,
          "per_class": true}'

Ausgabe wie ft_data.fetch: JSON-Zeilen, zuletzt {"type": "result", ...} oder "error".

Warum: Mehrere Quellen sollen in EIN Studio-Projekt mit EINER Klassenliste. Dafuer
muessen Klassen umbenannt, zusammengelegt oder weggelassen und Stichproben gezogen
werden – bevor der vorhandene Studio-Import (Ordner, Text, Audio, Videos) laeuft.
Das Ergebnis liegt in einer Form, die genau dieser Import liest:

  image_boxes                         images/ + labels/ (YOLO) + label_classes
  image_classify, audio_classify,
  video_classify                      <klasse>/<datei>
  audio_transcript                    <datei>.wav + <datei>.txt
  text_classify, text_pairs           data.jsonl mit text + label/target
"""
from __future__ import annotations

import csv
import io
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .detection import IMAGE_EXTS, convert_voc_to_yolo, label_path_for
from .fetch import (AUDIO_EXTS, VIDEO_EXTS, FetchError, coco_to_yolo, emit, inspect,
                    pair_images_and_labels, parquet_to_yolo, _read_class_list)

TEXT_COLS = ("text", "sentence", "content", "review", "document", "question", "input", "prompt", "source", "instruction")
LABEL_COLS = ("label", "labels", "class", "category", "sentiment", "intent", "topic", "target_label")
TARGET_COLS = ("target", "output", "answer", "response", "summary", "translation", "completion", "label_text")
TRANSCRIPT_COLS = ("transcription", "transcript", "sentence", "text", "normalized_text")
SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "dev", "test", "testing", "images", "labels", "data"}


class ClassMap:
    """Klassen umbenennen/zusammenlegen/weglassen. Schluessel ohne Gross/Klein."""

    def __init__(self, mapping: Optional[Dict[str, Optional[str]]], keep_unmapped: bool = True):
        self.m = {str(k).strip().lower(): (None if v is None or str(v).strip() == "" else str(v).strip())
                  for k, v in (mapping or {}).items()}
        self.keep = keep_unmapped
        self.dropped: Dict[str, int] = defaultdict(int)

    def __call__(self, name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        key = str(name).strip().lower()
        if key in self.m:
            out = self.m[key]
        else:
            out = str(name).strip() if self.keep else None
        if out is None:
            self.dropped[str(name)] += 1
        return out


def _sample(items: List[Tuple[Any, List[str]]], max_samples: int, per_class: bool, seed: int = 42) -> List[Tuple[Any, List[str]]]:
    """items: (eintrag, klassen-des-eintrags). per_class: gleich viele je Klasse (soweit vorhanden)."""
    rnd = random.Random(seed)
    items = list(items)
    rnd.shuffle(items)
    if not max_samples or len(items) <= max_samples:
        return items
    if not per_class:
        return items[:max_samples]
    by_cls: Dict[str, List[int]] = defaultdict(list)
    for i, (_, classes) in enumerate(items):
        for c in set(classes) or {"(none)"}:
            by_cls[c].append(i)
    chosen: List[int] = []
    seen = set()
    queues = {c: list(v) for c, v in by_cls.items()}
    while len(chosen) < max_samples and any(queues.values()):
        for c in sorted(queues, key=lambda k: len(by_cls[k])):  # seltene Klassen zuerst
            while queues[c]:
                i = queues[c].pop(0)
                if i not in seen:
                    seen.add(i)
                    chosen.append(i)
                    break
            if len(chosen) >= max_samples:
                break
    return [items[i] for i in chosen]


def _counts(items: Iterable[Tuple[Any, List[str]]]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for _, classes in items:
        for c in classes:
            out[c] += 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ── Objekterkennung ─────────────────────────────────────────────────────────

def _yolo_root(src: Path, work: Path) -> Path:
    rep = inspect(src)
    if rep["images"] == 0 and rep["parquet_files"]:
        out = work / "_yolo"
        parquet_to_yolo(src, out)
        return out
    if rep["format"] == "coco":
        out = coco_to_yolo(src)
        if out:
            return out
    if rep["format"] == "voc":
        copy = work / "_voc"
        shutil.copytree(src, copy)
        convert_voc_to_yolo(copy)
        return copy
    paired = pair_images_and_labels(src) if rep["images"] and not rep["yolo_labels"] else None
    return paired or src


def stage_boxes(src: Path, out: Path, cmap: ClassMap, max_samples: int, per_class: bool, keep_empty: bool) -> Dict[str, Any]:
    root = _yolo_root(src, out.parent)
    names = _read_class_list(root)
    images = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise FetchError("No images found for an object detection project.")
    target_names: List[str] = []
    items: List[Tuple[Any, List[str]]] = []
    for img in images:
        lp = label_path_for(img)
        lines_out: List[Tuple[str, str]] = []
        had_boxes = False
        if lp.exists():
            for line in lp.read_text(errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 5 or not parts[0].isdigit():
                    continue
                had_boxes = True
                cid = int(parts[0])
                src_name = names[cid] if cid < len(names) else f"class_{cid}"
                dst = cmap(src_name)
                if dst is None:
                    continue
                if dst not in target_names:
                    target_names.append(dst)
                lines_out.append((dst, " ".join(parts[1:5])))
        if not lines_out and had_boxes and not keep_empty:
            continue  # nur weggefilterte Klassen drauf: Bild faellt raus
        items.append(((img, lines_out), [d for d, _ in lines_out]))
    picked = _sample(items, max_samples, per_class)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for i, ((img, lines), _) in enumerate(picked):
        name = f"{i:06d}_{img.name}"
        shutil.copy2(img, out / "images" / name)
        (out / "labels" / (Path(name).stem + ".txt")).write_text(
            "".join(f"{target_names.index(d)} {coords}\n" for d, coords in lines))
    return {"studio_source": "folder", "path": str(out), "label_classes": target_names,
            "samples": len(picked), "per_class": _counts(picked)}


# ── Klassifikation (Bild, Audio, Video) ─────────────────────────────────────

def _class_names_for(schema, column: str) -> List[str]:
    try:
        meta = json.loads((schema.metadata or {}).get(b"huggingface", b"{}"))
        feat = meta.get("info", {}).get("features", {}).get(column, {})
        return list(feat.get("names") or [])
    except Exception:
        return []


def _media_from_parquet(src: Path, media_col_names: Tuple[str, ...], label_cols: Tuple[str, ...],
                        work: Path, ext_default: str) -> List[Tuple[Path, Optional[str], Optional[str]]]:
    """(datei, label, text) aus Parquet mit eingebetteten Medien."""
    import pyarrow.parquet as pq
    out: List[Tuple[Path, Optional[str], Optional[str]]] = []
    work.mkdir(parents=True, exist_ok=True)
    n = 0
    for pf in sorted(src.rglob("*.parquet")):
        f = pq.ParquetFile(pf)
        cols = f.schema_arrow.names
        mcol = next((c for c in media_col_names if c in cols), None)
        if not mcol:
            continue
        lcol = next((c for c in label_cols if c in cols), None)
        tcol = next((c for c in TRANSCRIPT_COLS if c in cols and c != lcol), None)
        names = _class_names_for(f.schema_arrow, lcol) if lcol else []
        for batch in f.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                m = row.get(mcol)
                data = m.get("bytes") if isinstance(m, dict) else m if isinstance(m, (bytes, bytearray)) else None
                if not data:
                    continue
                orig = (m.get("path") if isinstance(m, dict) else None) or ""
                ext = Path(orig).suffix.lower() or ext_default
                path = work / f"{n:07d}{ext}"
                path.write_bytes(data)
                lab = row.get(lcol) if lcol else None
                if isinstance(lab, int) and names and 0 <= lab < len(names):
                    lab = names[lab]
                out.append((path, None if lab is None else str(lab), row.get(tcol) if tcol else None))
                n += 1
    return out


def _class_folder_items(src: Path, exts: set) -> List[Tuple[Path, str]]:
    out = []
    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        cls = next((p for p in reversed(f.relative_to(src).parts[:-1]) if p.lower() not in SPLIT_NAMES), None)
        if cls:
            out.append((f, cls))
    return out


def stage_classify(src: Path, out: Path, modality: str, cmap: ClassMap, max_samples: int, per_class: bool) -> Dict[str, Any]:
    exts = {"image": IMAGE_EXTS, "audio": AUDIO_EXTS, "video": VIDEO_EXTS}[modality]
    pairs = _class_folder_items(src, exts)
    if not pairs and modality in ("image", "audio"):
        media_cols = ("image", "img") if modality == "image" else ("audio",)
        pairs = [(p, l) for p, l, _ in _media_from_parquet(src, media_cols, LABEL_COLS, out.parent / "_media",
                                                            ".jpg" if modality == "image" else ".wav") if l is not None]
    if not pairs:
        raise FetchError(f"No labeled {modality} files found (class folders or parquet with a label column).")
    items = []
    for f, cls in pairs:
        dst = cmap(cls)
        if dst is not None:
            items.append((f, [dst]))
    picked = _sample(items, max_samples, per_class)
    for i, (f, (cls,)) in enumerate(picked):
        d = out / _safe_name(cls)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, d / f"{i:06d}_{f.name}")
    if modality == "video":
        return {"studio_source": "videos", "paths": [str(p) for p in sorted(out.rglob("*")) if p.is_file()],
                "path": str(out), "samples": len(picked), "per_class": _counts(picked)}
    return {"studio_source": "folder" if modality == "image" else "audio", "path": str(out),
            "samples": len(picked), "per_class": _counts(picked)}


def _safe_name(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_." else "_" for c in name).strip(" .")
    return keep or "class"


# ── Audio mit Transkript ────────────────────────────────────────────────────

def stage_transcripts(src: Path, out: Path, max_samples: int) -> Dict[str, Any]:
    from .asr import resolve_asr_layout
    items: List[Tuple[Path, str]] = []
    try:
        layout = resolve_asr_layout(src)
        for audio, text in layout.train + layout.val + layout.test:  # AsrItem = (pfad, transkript)
            items.append((Path(audio), text))
    except Exception:
        items = []
    if not items:
        for f in sorted(src.rglob("*")):
            if f.suffix.lower() in AUDIO_EXTS and f.with_suffix(".txt").exists():
                items.append((f, f.with_suffix(".txt").read_text(errors="ignore").strip()))
    if not items:
        items = [(p, t) for p, _, t in _media_from_parquet(src, ("audio",), (), out.parent / "_media", ".wav") if t]
    if not items:
        raise FetchError("No audio with transcripts found.")
    picked = _sample([((a, t), []) for a, t in items], max_samples, False)
    out.mkdir(parents=True, exist_ok=True)
    for i, ((audio, text), _) in enumerate(picked):
        name = f"{i:06d}{audio.suffix.lower()}"
        shutil.copy2(audio, out / name)
        (out / (Path(name).stem + ".txt")).write_text(str(text), encoding="utf-8")
    return {"studio_source": "audio", "path": str(out), "samples": len(picked)}


# ── Text ─────────────────────────────────────────────────────────────────────

def _read_rows(f: Path, limit: int = 2_000_000) -> List[Dict[str, Any]]:
    ext = f.suffix.lower()
    if ext == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(f)
        rows = []
        for b in pf.iter_batches(batch_size=4096):
            rows.extend(b.to_pylist())
            if len(rows) >= limit:
                break
        for col in pf.schema_arrow.names:
            names = _class_names_for(pf.schema_arrow, col)
            if names:
                for r in rows:
                    if isinstance(r.get(col), int) and 0 <= r[col] < len(names):
                        r[col] = names[r[col]]
        return rows
    if ext in (".jsonl", ".json"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        if ext == ".json" and text.lstrip().startswith("["):
            return json.loads(text)[:limit]
        return [json.loads(l) for l in text.splitlines()[:limit] if l.strip()]
    if ext in (".csv", ".tsv"):
        with open(f, encoding="utf-8", errors="ignore", newline="") as fh:
            return list(csv.DictReader(fh, delimiter="\t" if ext == ".tsv" else ","))[:limit]
    return []


def _pick(cols: Iterable[str], candidates: Tuple[str, ...], override: Optional[str] = None) -> Optional[str]:
    cols = list(cols)
    if override and override in cols:
        return override
    low = {c.lower(): c for c in cols}
    return next((low[c] for c in candidates if c in low), None)


def stage_text(src: Path, out: Path, pairs: bool, cmap: ClassMap, max_samples: int, per_class: bool,
               text_col: Optional[str], label_col: Optional[str]) -> Dict[str, Any]:
    files = [f for f in sorted(src.rglob("*")) if f.suffix.lower() in (".parquet", ".jsonl", ".json", ".csv", ".tsv")]
    rows: List[Dict[str, Any]] = []
    for f in files:
        rows.extend(_read_rows(f))
    if not rows:
        raise FetchError("No table rows found (parquet, CSV, TSV, JSON or JSONL).")
    cols = list(rows[0].keys())
    items = []
    notes: List[str] = []
    source_columns: Dict[str, Any] = {}
    if pairs and not (text_col and label_col):
        # LLM- und Seq2Seq-Formate wie beim Training lesen (ft_data.llm): Chat/ShareGPT,
        # Alpaca (Anweisung + Eingabe), prompt/completion, Frage/Antwort, Praeferenzen.
        from . import llm
        try:
            fmt, fcols = llm.detect_format(cols)
        except ValueError as e:
            raise FetchError(str(e))
        if fmt == "text":
            raise FetchError(f"Rows only have a text column ({fcols[0]}) and no answer. Use a text classification "
                             "project with a label column, or train the LLM on this dataset directly (continued pretraining).")
        source_columns = {"format": fmt, "columns": list(c for c in fcols if c)}
        if fmt == "preference":
            notes.append("Preference data: the chosen answer becomes the target, the rejected answer is not kept.")
        for r in rows:
            ex = llm.row_to_example(r, fmt, fcols)
            if ex is None or not ex.messages:
                continue
            prompt = _render_prompt(ex.prompt_messages())
            answer = ex.reference()
            if prompt and answer:
                items.append(((prompt, answer), []))
    else:
        tcol = _pick(cols, TEXT_COLS, text_col)
        lcol = _pick(cols, TARGET_COLS if pairs else LABEL_COLS, label_col)
        if not tcol or not lcol:
            raise FetchError(f"Could not find the {'input/target' if pairs else 'text/label'} columns. Columns: {cols[:20]}. "
                             "Pass text_column and label_column.")
        source_columns = {"text": tcol, ("target" if pairs else "label"): lcol}
        for r in rows:
            text, lab = r.get(tcol), r.get(lcol)
            if text is None or lab is None or str(text).strip() == "":
                continue
            if pairs:
                items.append(((str(text), str(lab)), []))
            else:
                dst = cmap(str(lab))
                if dst is not None:
                    items.append(((str(text), dst), [dst]))
    if not items:
        raise FetchError("No usable rows after reading the table.")
    picked = _sample(items, max_samples, per_class and not pairs)
    out.mkdir(parents=True, exist_ok=True)
    key = "target" if pairs else "label"
    with open(out / "data.jsonl", "w", encoding="utf-8") as fh:
        for (text, lab), _ in picked:
            fh.write(json.dumps({"text": text, key: lab}, ensure_ascii=False) + "\n")
    res = {"studio_source": "text", "path": str(out / "data.jsonl"), "text_column": "text", "label_column": key,
           "samples": len(picked), "source_columns": source_columns}
    if notes:
        res["notes"] = notes
    if not pairs:
        res["per_class"] = _counts(picked)
    return res


def _render_prompt(msgs: List[Dict[str, str]]) -> str:
    """Prompt-Verlauf als Text. Eine einzelne Nutzer-Nachricht bleibt, wie sie ist."""
    if len(msgs) == 1 and msgs[0]["role"] == "user":
        return msgs[0]["content"]
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    return "\n\n".join(f"{names.get(m['role'], m['role'])}: {m['content']}" for m in msgs)


# ── Einstieg ────────────────────────────────────────────────────────────────

def stage(req: Dict[str, Any]) -> Dict[str, Any]:
    src = Path(req["source"])
    out = Path(req["target"]) / "staged"
    if out.exists():
        shutil.rmtree(out)
    kind = req["project_kind"]
    cmap = ClassMap(req.get("class_map"), bool(req.get("keep_unmapped", True)))
    n = int(req.get("max_samples") or 0)
    per = bool(req.get("per_class", False))
    if kind == "image_boxes":
        res = stage_boxes(src, out, cmap, n, per, bool(req.get("keep_empty", False)))
    elif kind in ("image_classify", "audio_classify", "video_classify"):
        res = stage_classify(src, out, kind.split("_")[0], cmap, n, per)
    elif kind == "audio_transcript":
        res = stage_transcripts(src, out, n)
    elif kind in ("text_classify", "text_pairs"):
        res = stage_text(src, out, kind == "text_pairs", cmap, n, per, req.get("text_column"), req.get("label_column"))
    else:
        raise FetchError(f"Unknown project kind: {kind}")
    if cmap.dropped:
        res["dropped_classes"] = dict(sorted(cmap.dropped.items(), key=lambda kv: -kv[1])[:30])
    return res


def main(argv: List[str]) -> int:
    try:
        req = json.loads(argv[1]) if len(argv) > 1 else json.load(sys.stdin)
        emit("result", **stage(req))
        return 0
    except FetchError as e:
        emit("error", message=str(e))
        return 2
    except Exception as e:  # noqa: BLE001
        emit("error", message=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
