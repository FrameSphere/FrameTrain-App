"""Datensaetze holen und pruefen – fuer die KI-Werkzeuge (MCP) und die App.

Aufruf (eine JSON-Anfrage, Ausgabe JSON-Zeilen auf stdout):

    python -m ft_data.fetch '{"kind": "hf_files", "repo_id": "..."}'
    python -m ft_data.fetch '{"kind": "hf", "repo_id": "...", "target": "...", "include": ["sample/"], "max_mb": 2000}'
    python -m ft_data.fetch '{"kind": "url", "url": "https://.../data.zip", "target": "..."}'
    python -m ft_data.fetch '{"kind": "inspect", "path": "..."}'

Zeilen: {"type": "status"|"progress"|"result"|"error", ...}. Die letzte Zeile ist
"result" (Erfolg) oder "error".

Warum: Hugging-Face-Datensaetze fuer Objekterkennung liegen oft als Parquet mit
eingebetteten Bildern vor. Der bisherige Import legte nur die Parquet-Dateien ab –
YOLO fand dann keine Bilder. Ausserdem liess sich kein Unterordner laden (ein
100-Bilder-Beispiel in einem 22-GB-Repo) und kein Archiv von einer URL.
"""
from __future__ import annotations

import fnmatch
import io
import json
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .detection import IMAGE_EXTS

DEFAULT_MAX_MB = 2000
HARD_MAX_MB = 20000
LABEL_EXTS = {".txt", ".xml", ".json"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def emit(kind: str, **data: Any) -> None:
    print(json.dumps({"type": kind, **data}, ensure_ascii=False), flush=True)


class FetchError(Exception):
    pass


# ── Pruefen ─────────────────────────────────────────────────────────────────

def inspect(path: Path) -> Dict[str, Any]:
    """Was liegt im Ordner? Bilder, YOLO-Labels, VOC, COCO, Parquet, Klassen, Splits."""
    path = Path(path)
    images = yolo = voc = coco = parquet = csv = audio = video = transcripts = 0
    class_dirs: Dict[str, int] = {}
    splits: Dict[str, int] = {}
    yaml_files: List[str] = []
    for f in path.rglob("*"):
        if not f.is_file() or f.name.startswith(".") or ".frametrain_media" in f.parts:
            continue
        ext = f.suffix.lower()
        rel_parts = [p.lower() for p in f.relative_to(path).parts]
        if ext in IMAGE_EXTS:
            images += 1
            if "labels" not in rel_parts and "images" not in rel_parts:
                class_dirs[f.parent.name] = class_dirs.get(f.parent.name, 0) + 1
            for s in ("train", "val", "valid", "validation", "test"):
                if s in rel_parts:
                    key = "val" if s in ("valid", "validation") else s
                    splits[key] = splits.get(key, 0) + 1
                    break
        elif ext in AUDIO_EXTS:
            audio += 1
            if f.with_suffix(".txt").exists():
                transcripts += 1
            class_dirs[f.parent.name] = class_dirs.get(f.parent.name, 0) + 1
        elif ext in VIDEO_EXTS:
            video += 1
            class_dirs[f.parent.name] = class_dirs.get(f.parent.name, 0) + 1
        elif ext == ".txt" and "labels" in rel_parts:
            yolo += 1
        elif ext == ".xml":
            voc += 1
        elif ext == ".json" and ("annotations" in rel_parts or "coco" in f.name.lower() or f.name.startswith("_annotations")):
            coco += 1
        elif ext == ".parquet":
            parquet += 1
        elif ext in (".csv", ".jsonl", ".tsv"):
            csv += 1
        elif ext in (".yaml", ".yml"):
            yaml_files.append(str(f.relative_to(path)))
    classes = _yaml_classes(path, yaml_files)
    folder_classes = sorted(k for k in class_dirs if k.lower() not in
                            ("train", "val", "valid", "validation", "test", "images", "audio", "clips", "videos", path.name.lower()))
    if images and yolo:
        fmt = "yolo"
    elif images and voc:
        fmt = "voc"
    elif images and coco:
        fmt = "coco"
    elif images:
        fmt = "image_folders" if len(folder_classes) > 1 else "images"
    elif audio:
        fmt = "audio_transcript" if transcripts else ("audio_folders" if len(folder_classes) > 1 else "audio")
    elif video:
        fmt = "video_folders" if len(folder_classes) > 1 else "video"
    elif parquet:
        fmt = "parquet"
    elif csv:
        fmt = "table"
    else:
        fmt = "unknown"
    if not classes and fmt.endswith("_folders"):
        classes = folder_classes
    problems: List[str] = []
    if fmt == "unknown":
        problems.append("No recognizable data files (images, audio, video, parquet, CSV/JSONL) found.")
    if fmt == "images":
        problems.append("Images without labels or class folders. Usable for labeling in the Dataset Builder, not directly for supervised training.")
    if fmt == "yolo" and yolo < images * 0.5:
        problems.append(f"Only {yolo} label files for {images} images. Many images are unlabeled.")
    if fmt == "audio" and not transcripts:
        problems.append("Audio without transcripts or class folders.")
    table = table_summary(path) if fmt in ("table", "parquet") else None
    if table and table.get("problems"):
        problems.extend(table["problems"])
    return {
        "table": table,
        "format": fmt, "images": images, "audio": audio, "video": video, "transcripts": transcripts, "yolo_labels": yolo, "voc_xml": voc, "coco_json": coco,
        "parquet_files": parquet, "table_files": csv, "splits": splits, "classes": classes,
        "yaml_files": yaml_files[:5], "problems": problems,
    }


TABLE_SUFFIXES = (".parquet", ".csv", ".tsv", ".jsonl", ".json")
LABEL_CANDIDATES = ("label", "labels", "class", "category", "sentiment", "intent", "topic", "target_label")
TEXT_CANDIDATES = ("text", "sentence", "content", "review", "document", "question", "prompt", "instruction", "input")


def _clip(v: Any, n: int = 200) -> Any:
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, dict) and "bytes" in v:
        return "<media>"
    t = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return t if len(t) <= n else t[:n] + " ..."


def table_summary(root: Path, max_rows: int = 50_000) -> Optional[Dict[str, Any]]:
    """Spalten, Zeilen, erkannte Aufgabe, Label-Verteilung und Beispielzeilen einer Tabelle."""
    files = [f for f in sorted(root.rglob("*")) if f.is_file() and f.suffix.lower() in TABLE_SUFFIXES
             and not f.name.startswith(".")]
    if not files:
        return None
    total_rows = 0
    rows: List[Dict[str, Any]] = []
    columns: List[str] = []
    for f in files[:50]:
        try:
            if f.suffix.lower() == ".parquet":
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(f)
                total_rows += pf.metadata.num_rows
                if not columns:
                    columns = pf.schema_arrow.names
                if len(rows) < max_rows:
                    for b in pf.iter_batches(batch_size=2048, columns=[c for c in pf.schema_arrow.names]):
                        rows.extend(b.to_pylist())
                        if len(rows) >= max_rows:
                            break
                    names = {}
                    meta = json.loads((pf.schema_arrow.metadata or {}).get(b"huggingface", b"{}") or b"{}")
                    for col, feat in (meta.get("info", {}).get("features", {}) or {}).items():
                        if isinstance(feat, dict) and feat.get("names"):
                            names[col] = feat["names"]
                    for r in rows:
                        for col, nm in names.items():
                            if isinstance(r.get(col), int) and 0 <= r[col] < len(nm):
                                r[col] = nm[r[col]]
            else:
                from .stage import _read_rows
                part = _read_rows(f, limit=max_rows)
                total_rows += len(part)
                if not columns and part:
                    columns = list(part[0].keys())
                rows.extend(part[: max(0, max_rows - len(rows))])
        except Exception as e:  # noqa: BLE001 – eine kaputte Datei soll den Bericht nicht kippen
            return {"files": len(files), "problems": [f"Could not read {f.name}: {e}"]}
    low = {c.lower(): c for c in columns}
    task = "unknown"
    detected: Dict[str, Any] = {}
    try:
        from . import llm
        fmt, cols = llm.detect_format(columns)
        if fmt in ("messages", "pair", "preference"):
            task = {"messages": "chat", "pair": "pairs", "preference": "preference"}[fmt]
            detected = {"llm_format": fmt, "columns": [c for c in cols if c]}
    except Exception:
        pass
    label_col = next((low[c] for c in LABEL_CANDIDATES if c in low), None)
    text_col = next((low[c] for c in TEXT_CANDIDATES if c in low), None)
    labels: Dict[str, int] = {}
    if task == "unknown" and label_col and text_col:
        task = "classification"
        detected = {"text_column": text_col, "label_column": label_col}
        for r in rows:
            v = r.get(label_col)
            if v is not None:
                labels[str(v)] = labels.get(str(v), 0) + 1
    elif task == "unknown" and text_col:
        task = "text"
        detected = {"text_column": text_col}
    problems: List[str] = []
    if labels:
        counts = sorted(labels.values())
        if len(counts) > 1 and counts[-1] > 10 * max(1, counts[0]):
            problems.append(f"Labels are very unbalanced ({counts[-1]} vs {counts[0]} rows).")
    empty = sum(1 for r in rows[:5000] if text_col and not str(r.get(text_col) or "").strip())
    if empty:
        problems.append(f"{empty} of the first {min(len(rows), 5000)} rows have an empty {text_col}.")
    return {
        "files": len(files), "rows": total_rows, "columns": columns[:40], "task": task, "detected": detected,
        "labels": dict(sorted(labels.items(), key=lambda kv: -kv[1])[:30]),
        "label_count": len(labels), "counted_rows": len(rows),
        "examples": [{k: _clip(v) for k, v in list(r.items())[:8]} for r in rows[:3]],
        "problems": problems,
    }


def _yaml_classes(root: Path, yaml_files: List[str]) -> List[str]:
    for rel in yaml_files:
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except OSError:
            continue
        names = _parse_names(text)
        if names:
            return names
    return []


def _parse_names(text: str) -> List[str]:
    """names: [a, b] | names:\\n  - a | names:\\n  0: a – ohne PyYAML-Abhaengigkeit."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith("names:"):
            continue
        rest = s[len("names:"):].strip()
        if rest.startswith("["):
            return [x.strip().strip("'\"") for x in rest.strip("[]").split(",") if x.strip()]
        out: List[str] = []
        for nxt in lines[i + 1:]:
            t = nxt.strip()
            if not t or t.startswith("#"):
                continue
            if not nxt.startswith((" ", "\t")):
                break
            if t.startswith("- "):
                out.append(t[2:].strip().strip("'\""))
            elif ":" in t:
                out.append(t.split(":", 1)[1].strip().strip("'\""))
        return out
    return []


# ── Hugging Face ────────────────────────────────────────────────────────────

def _api():
    from huggingface_hub import HfApi
    return HfApi()


def hf_list(repo_id: str, revision: Optional[str] = None) -> List[Tuple[str, int]]:
    """(Pfad, Bytes) aller Dateien eines Dataset-Repos."""
    out: List[Tuple[str, int]] = []
    for item in _api().list_repo_tree(repo_id, repo_type="dataset", recursive=True, revision=revision):
        size = getattr(item, "size", None)
        if size is not None and not hasattr(item, "tree_id"):
            out.append((item.path, int(size)))
    return out


def summarize(files: List[Tuple[str, int]]) -> Dict[str, Any]:
    folders: Dict[str, Dict[str, int]] = {}
    exts: Dict[str, int] = {}
    total = 0
    for p, size in files:
        total += size
        top = p.split("/", 1)[0] if "/" in p else "(root)"
        f = folders.setdefault(top, {"files": 0, "bytes": 0, "images": 0})
        f["files"] += 1
        f["bytes"] += size
        ext = Path(p).suffix.lower()
        if ext in IMAGE_EXTS:
            f["images"] += 1
        exts[ext or "(none)"] = exts.get(ext or "(none)", 0) + 1
    biggest = sorted(folders.items(), key=lambda kv: -kv[1]["bytes"])[:25]
    return {
        "files": len(files), "total_mb": round(total / 1e6, 1),
        "folders": [{"path": k, **v, "mb": round(v["bytes"] / 1e6, 1)} for k, v in biggest],
        "extensions": dict(sorted(exts.items(), key=lambda kv: -kv[1])[:15]),
        "has_parquet": exts.get(".parquet", 0) > 0,
        "image_files": sum(v for k, v in exts.items() if k in IMAGE_EXTS),
    }


def select(files: List[Tuple[str, int]], include: Iterable[str]) -> List[Tuple[str, int]]:
    """include: Ordner ("sample/"), Dateien oder Muster ("*/labels/*.txt")."""
    pats = [p.strip() for p in include if p and p.strip()]
    if not pats:
        return list(files)
    out = []
    for path, size in files:
        for pat in pats:
            if pat.endswith("/") and path.startswith(pat):
                out.append((path, size)); break
            if path == pat or fnmatch.fnmatch(path, pat) or path.startswith(pat.rstrip("/") + "/"):
                out.append((path, size)); break
    return out


def hf_files(req: Dict[str, Any]) -> Dict[str, Any]:
    repo = req["repo_id"]
    files = hf_list(repo)
    info = summarize(files)
    info["repo_id"] = repo
    if not info["image_files"] and not info["has_parquet"]:
        try:
            conv = hf_list(repo, revision="refs/convert/parquet")
            info["converted_parquet_mb"] = round(sum(s for _, s in conv) / 1e6, 1)
        except Exception:
            pass
    return info


def hf_download(req: Dict[str, Any], target: Path) -> Path:
    from huggingface_hub import hf_hub_download
    repo = req["repo_id"]
    max_mb = min(float(req.get("max_mb") or DEFAULT_MAX_MB), HARD_MAX_MB)
    files = hf_list(repo)
    chosen = select(files, req.get("include") or [])
    if req.get("include") and not chosen:
        raise FetchError(f"Nothing in {repo} matches {req['include']}. Use datasets_hf_files to see the folders.")
    revision = None
    has_media = any(Path(p).suffix.lower() in IMAGE_EXTS | LABEL_EXTS | {".parquet", ".csv", ".jsonl"} for p, _ in chosen)
    if not req.get("include") and not has_media:
        # Nur Skripte/README: die von Hugging Face konvertierten Parquet-Dateien nehmen
        revision = "refs/convert/parquet"
        chosen = [f for f in hf_list(repo, revision=revision) if f[0].endswith(".parquet")]
    if not req.get("include"):
        # Ohne Auswahl: README, Skripte, Videos weglassen – nur Daten
        keep = IMAGE_EXTS | LABEL_EXTS | {".parquet", ".csv", ".jsonl", ".tsv", ".yaml", ".yml"}
        chosen = [f for f in chosen if Path(f[0]).suffix.lower() in keep]
    total = sum(s for _, s in chosen)
    if total > max_mb * 1e6:
        summary = summarize(chosen)
        raise FetchError(
            f"Selection is {total / 1e6:.0f} MB, above the limit of {max_mb:.0f} MB. "
            f"Pick a folder with include, e.g. {[f['path'] + '/' for f in summary['folders'][:3]]}, or raise max_mb.")
    if not chosen:
        raise FetchError(f"No data files found in {repo}.")
    raw = target / "raw"
    done = 0
    for i, (path, size) in enumerate(chosen):
        hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=str(raw))
        done += size
        if i % 10 == 0 or i == len(chosen) - 1:
            emit("progress", file_index=i + 1, total_files=len(chosen), downloaded_bytes=done, total_bytes=total,
                 message=f"{i + 1}/{len(chosen)}: {path}")
    root = raw
    if req.get("include") and len(req["include"]) == 1:
        sub = raw / req["include"][0].rstrip("/")
        if sub.is_dir():
            root = sub
    return root


# ── Parquet mit Bildern -> YOLO ─────────────────────────────────────────────

def _bbox_to_yolo(b: List[float], w: float, h: float, fmt: str) -> Optional[Tuple[float, float, float, float]]:
    if len(b) != 4 or w <= 0 or h <= 0:
        return None
    x0, y0, a, c = (float(v) for v in b)
    if fmt == "yolo":
        cx, cy, bw, bh = x0, y0, a, c
    else:
        if fmt == "xyxy":
            x1, y1 = a, c
        else:  # coco: x, y, Breite, Hoehe (Hugging-Face-Standard)
            x1, y1 = x0 + a, y0 + c
        if max(x0, y0, x1, y1) <= 1.0001:  # bereits normalisiert
            w = h = 1.0
        cx, cy, bw, bh = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h
    if bw <= 0 or bh <= 0:
        return None
    clamp = lambda v: min(max(v, 0.0), 1.0)
    return clamp(cx), clamp(cy), clamp(bw), clamp(bh)


def _class_names_from_schema(schema) -> List[str]:
    """ClassLabel-Namen stehen in den Hugging-Face-Metadaten des Parquet-Schemas."""
    try:
        meta = schema.metadata or {}
        raw = meta.get(b"huggingface")
        if not raw:
            return []
        feats = json.loads(raw).get("info", {}).get("features", {})
        objs = feats.get("objects", {})
        cat = (objs.get("feature", objs) or {}).get("category") if isinstance(objs, dict) else None
        if isinstance(objs, list) and objs:
            cat = objs[0].get("category")
        if isinstance(cat, dict):
            names = cat.get("names") or (cat.get("feature") or {}).get("names")
            if names:
                return list(names)
    except Exception:
        pass
    return []


def _find_object_columns(row: Dict[str, Any]) -> Tuple[Optional[List], Optional[List]]:
    """bbox- und Kategorie-Listen aus "objects" (dict of lists) oder Top-Level-Spalten."""
    objs = row.get("objects") or row.get("annotations") or row.get("labels")
    if isinstance(objs, dict):
        boxes = objs.get("bbox") or objs.get("bboxes") or objs.get("boxes")
        cats = objs.get("category") or objs.get("categories") or objs.get("label") or objs.get("labels") or objs.get("category_id")
        return boxes, cats
    if isinstance(objs, list) and objs and isinstance(objs[0], dict):
        boxes = [o.get("bbox") for o in objs]
        cats = [o.get("category", o.get("category_id", o.get("label"))) for o in objs]
        return boxes, cats
    return row.get("bbox") or row.get("bboxes"), row.get("category") or row.get("categories")


def parquet_to_yolo(src: Path, out: Path, max_rows: int = 0, bbox_format: str = "coco") -> Dict[str, Any]:
    """Bilder und Boxen aus Hugging-Face-Parquet als YOLO-Ordner ablegen."""
    import pyarrow.parquet as pq
    from PIL import Image

    files = sorted(src.rglob("*.parquet"))
    names: List[str] = []
    written = boxes_total = skipped = 0
    seen_cats: Dict[Any, int] = {}
    for pf in files:
        split = "val" if any(s in pf.as_posix().lower() for s in ("valid", "/val", "-val", "test")) else "train"
        table = pq.ParquetFile(pf)
        if not names:
            names = _class_names_from_schema(table.schema_arrow)
        for batch in table.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                if max_rows and written >= max_rows:
                    break
                img = row.get("image") or row.get("img")
                data = img.get("bytes") if isinstance(img, dict) else img if isinstance(img, (bytes, bytearray)) else None
                if not data:
                    skipped += 1
                    continue
                try:
                    im = Image.open(io.BytesIO(data))
                    w, h = im.size
                    ext = ".png" if (im.format or "").upper() == "PNG" else ".jpg"
                except Exception:
                    skipped += 1
                    continue
                bxs, cats = _find_object_columns(row)
                lines = []
                for b, c in zip(bxs or [], cats or []):
                    yb = _bbox_to_yolo(b, w, h, bbox_format)
                    if yb is None or c is None:
                        continue
                    if isinstance(c, str):
                        if c not in seen_cats:
                            seen_cats[c] = len(seen_cats)
                        cid = seen_cats[c]
                    else:
                        cid = int(c)
                    lines.append(f"{cid} {yb[0]:.6f} {yb[1]:.6f} {yb[2]:.6f} {yb[3]:.6f}")
                stem = f"{written:06d}"
                (out / split / "images").mkdir(parents=True, exist_ok=True)
                (out / split / "labels").mkdir(parents=True, exist_ok=True)
                (out / split / "images" / f"{stem}{ext}").write_bytes(data)
                (out / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                written += 1
                boxes_total += len(lines)
                if written % 50 == 0:
                    emit("status", message=f"Converted {written} images to YOLO")
            if max_rows and written >= max_rows:
                break
        if max_rows and written >= max_rows:
            break
    if written == 0:
        raise FetchError("The parquet files contain no images (no 'image' column with bytes). This dataset is not usable for detection.")
    if not names and seen_cats:
        names = [k for k, _ in sorted(seen_cats.items(), key=lambda kv: kv[1])]
    if not names:
        top = 0
        for lf in out.rglob("*.txt"):
            for line in lf.read_text().splitlines():
                if line.strip():
                    top = max(top, int(line.split()[0]))
        names = [f"class_{i}" for i in range(top + 1)]
    if not (out / "val" / "images").is_dir():
        _move_share_to_val(out, 0.15)
    yaml = ["path: .", "train: train/images", "val: val/images", f"nc: {len(names)}", "names:"]
    yaml += [f"  - '{n}'" for n in names]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    return {"images": written, "boxes": boxes_total, "skipped_rows": skipped, "classes": names}


def _move_share_to_val(out: Path, share: float) -> None:
    imgs = sorted((out / "train" / "images").iterdir())
    n = max(1, int(len(imgs) * share)) if len(imgs) > 5 else 0
    (out / "val" / "images").mkdir(parents=True, exist_ok=True)
    (out / "val" / "labels").mkdir(parents=True, exist_ok=True)
    for img in imgs[-n:] if n else []:
        lbl = out / "train" / "labels" / (img.stem + ".txt")
        shutil.move(str(img), out / "val" / "images" / img.name)
        if lbl.exists():
            shutil.move(str(lbl), out / "val" / "labels" / lbl.name)


def needs_yolo_conversion(root: Path) -> bool:
    rep = inspect(root)
    return rep["images"] == 0 and rep["parquet_files"] > 0


# ── URL / Archiv ────────────────────────────────────────────────────────────

MEDIA_EXTS = IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS
SIDECAR_MAX = 50 * 1024 * 1024  # Begleitdateien (yaml, json, csv) bis 50 MB immer mitnehmen


class RangeFile(io.RawIOBase):
    """Datei auf einem https-Server, gelesen per HTTP-Range – damit zipfile nur das
    Inhaltsverzeichnis und die gewaehlten Eintraege laedt statt des ganzen Archivs."""

    def __init__(self, url: str, size: int, session):
        self.url, self.size, self.s, self.pos, self.fetched = url, size, session, 0, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        if self.pos >= self.size:
            return b""
        end = self.size - 1 if n is None or n < 0 else min(self.size - 1, self.pos + n - 1)
        r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "FrameTrain-Desktop"}, timeout=60)
        if r.status_code != 206:
            raise FetchError("The server does not support partial downloads.")
        data = r.content
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def _head(session, url: str) -> Tuple[int, bool]:
    r = session.head(url, allow_redirects=True, timeout=30, headers={"User-Agent": "FrameTrain-Desktop"})
    size = int(r.headers.get("content-length") or 0)
    ranges = r.headers.get("accept-ranges", "").lower() == "bytes"
    return size, ranges


def _matches(name: str, include: List[str]) -> bool:
    if not include:
        return True
    return bool(select([(name, 0)], include))


def partial_zip(url: str, out: Path, include: List[str], max_files: int, stems: Optional[set], session, max_mb: float) -> Dict[str, Any]:
    """Nur ausgewaehlte Eintraege eines entfernten ZIPs entpacken."""
    import requests  # noqa: F401 – session kommt von requests
    size, _ = _head(session, url)
    rf = RangeFile(url, size, session)
    taken = media = 0
    with zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20)) as z:
        entries = [m for m in z.infolist() if not m.is_dir() and "__MACOSX" not in m.filename
                   and _matches(m.filename, include)]

        def is_media(m) -> bool:
            return Path(m.filename).suffix.lower() in MEDIA_EXTS

        def is_sidecar(m) -> bool:
            ext = Path(m.filename).suffix.lower()
            return (not is_media(m) and m.file_size <= SIDECAR_MAX
                    and (ext in (".yaml", ".yml", ".names", ".csv", ".tsv", ".jsonl", ".json")
                         or Path(m.filename).name in ("classes.txt", "labels.txt", "obj.names")))

        if stems is None and max_files:
            # Erstes Archiv mit Limit: Medien waehlen, dann nur die dazu passenden Labels
            chosen_media = [m for m in entries if is_media(m)][:max_files]
            keep = {Path(m.filename).stem for m in chosen_media}
            selected = chosen_media + [m for m in entries if not is_media(m)
                                       and (Path(m.filename).stem in keep or is_sidecar(m))]
        elif stems is not None:
            # Weiteres Archiv (z. B. Labels): nur, was zu den gewaehlten Medien passt
            selected = [m for m in entries if Path(m.filename).stem in stems or is_sidecar(m)]
        else:
            selected = entries
        for m in selected:
            if rf.fetched > max_mb * 1e6:
                raise FetchError(f"Partial download exceeded {max_mb:.0f} MB.")
            dest = _safe_target(out, m.filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(dest, "wb") as fh:
                shutil.copyfileobj(src, fh)
            taken += 1
            media += 1 if is_media(m) else 0
            if taken % 25 == 0:
                emit("progress", downloaded_bytes=rf.fetched, total_bytes=size, message=f"{taken} file(s) from {Path(url).name}")
    return {"files": taken, "media": media, "downloaded_mb": round(rf.fetched / 1e6, 1), "archive_mb": round(size / 1e6, 1)}


def url_download(req: Dict[str, Any], target: Path) -> Path:
    """Eine oder mehrere https-Quellen in einen Ordner holen.

    urls: Liste (oder url: einzeln). include: Muster fuer Archiv-Eintraege.
    max_files: hoechstens so viele Medien-Dateien aus dem ersten Archiv; weitere
    Archive (z. B. die Labels) liefern dann nur die passenden Dateinamen.
    Grosse ZIPs werden – wenn der Server es kann – nur teilweise geladen.
    """
    import requests
    urls = [str(u) for u in (req.get("urls") or ([req["url"]] if req.get("url") else []))]
    if not urls:
        raise FetchError("No URL given.")
    for u in urls:
        if not u.lower().startswith("https://"):
            raise FetchError("Only https:// URLs are allowed.")
    max_mb = min(float(req.get("max_mb") or DEFAULT_MAX_MB), HARD_MAX_MB)
    include = [p for p in (req.get("include") or []) if p]
    max_files = int(req.get("max_files") or 0)
    out = target / "extracted"
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    stems: Optional[set] = None
    notes: List[str] = []
    for u in urls:
        size, ranges = _head(session, u)
        name = u.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] or "download"
        is_zip = name.lower().endswith(".zip")
        want_partial = is_zip and ranges and (include or max_files or stems is not None or size > max_mb * 1e6)
        if want_partial:
            emit("status", message=f"Reading the table of contents of {name} ({size / 1e6:.0f} MB)")
            before = {p for p in out.rglob("*") if p.is_file()}
            info = partial_zip(u, out, include, max_files, stems, session, max_mb)
            notes.append(f"{name}: {info['files']} file(s), {info['downloaded_mb']} MB of {info['archive_mb']} MB loaded")
            if stems is None and info["media"]:
                stems = {p.stem for p in out.rglob("*") if p.is_file() and p not in before and p.suffix.lower() in MEDIA_EXTS}
            continue
        if size and size > max_mb * 1e6:
            hint = " The server does not allow partial downloads." if is_zip and not ranges else ""
            raise FetchError(f"{name} is {size / 1e6:.0f} MB, above the limit of {max_mb:.0f} MB.{hint}")
        dest = target / name
        with session.get(u, stream=True, timeout=60, headers={"User-Agent": "FrameTrain-Desktop"}) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length") or 0)
            done = 0
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if done > max_mb * 1e6:
                        raise FetchError(f"Download exceeded the limit of {max_mb:.0f} MB.")
                    emit("progress", downloaded_bytes=done, total_bytes=total, message=name)
        sub = out / Path(name).stem.replace(".tar", "") if len(urls) > 1 else out
        if zipfile.is_zipfile(dest) or tarfile.is_tarfile(dest):
            extract_if_archive(dest, sub)
        else:
            sub.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), sub / name)
        notes.append(f"{name}: {total / 1e6:.1f} MB loaded")
    if notes:
        emit("status", message="; ".join(notes))
    return _descend_single_folder(out)


def extract_if_archive(path: Path, out: Path) -> Path:
    """ZIP/TAR sicher entpacken (keine Pfade ausserhalb von out). Liefert den Datenordner."""
    lower = path.name.lower()
    if zipfile.is_zipfile(path):
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            for m in z.infolist():
                _safe_target(out, m.filename)
            z.extractall(out)
    elif lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")) or tarfile.is_tarfile(path):
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path) as t:
            members = []
            for m in t.getmembers():
                if m.issym() or m.islnk() or m.isdev():
                    continue
                _safe_target(out, m.name)
                members.append(m)
            t.extractall(out, members=members)
    else:
        return path.parent
    path.unlink(missing_ok=True)
    return _descend_single_folder(out)


def _safe_target(root: Path, member: str) -> Path:
    dest = (root / member).resolve()
    if not str(dest).startswith(str(root.resolve()) + os.sep) and dest != root.resolve():
        raise FetchError(f"Archive entry outside the target folder: {member}")
    return dest


def _descend_single_folder(root: Path) -> Path:
    entries = [e for e in root.iterdir() if not e.name.startswith(".") and e.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return root


# ── Formate angleichen ──────────────────────────────────────────────────────

def _read_class_list(root: Path) -> List[str]:
    for name in ("classes.txt", "obj.names", "labels.txt"):
        for f in root.rglob(name):
            names = [l.strip() for l in f.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
            if names:
                return names
    for f in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
        names = _parse_names(f.read_text(encoding="utf-8", errors="ignore"))
        if names:
            return names
    for f in root.rglob("notes.json"):  # Label-Studio-Export
        try:
            cats = json.loads(f.read_text()).get("categories", [])
            if cats:
                return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]
        except Exception:
            pass
    return []


def pair_images_and_labels(root: Path) -> Optional[Path]:
    """Bilder und YOLO-Labels liegen in getrennten Baeumen (z. B. aus zwei Archiven):
    nach Dateinamen paaren und als images/ + labels/ ablegen."""
    from .detection import label_path_for
    images = {p.stem: p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    if not images:
        return None
    if any(label_path_for(p).exists() for p in list(images.values())[:50]):
        return None  # Ultralytics findet die Labels schon
    labels = {p.stem: p for p in root.rglob("*.txt")
              if p.stem in images and p.name not in ("classes.txt", "labels.txt", "README.txt")}
    if not labels:
        return None
    out = root.parent / (root.name + "_paired")
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
    stems = sorted(set(images) & set(labels))
    n_val = max(1, int(len(stems) * 0.15)) if len(stems) > 5 else 0
    for i, stem in enumerate(stems):
        split = "val" if i >= len(stems) - n_val else "train"
        img = images[stem]
        shutil.copy2(img, out / split / "images" / img.name)
        shutil.copy2(labels[stem], out / split / "labels" / f"{stem}.txt")
    names = _read_class_list(root)
    _write_data_yaml(out, names or _names_from_labels(out))
    return out


def _names_from_labels(out: Path) -> List[str]:
    top = -1
    for lf in out.rglob("*.txt"):
        for line in lf.read_text(errors="ignore").splitlines():
            parts = line.split()
            if parts and parts[0].isdigit():
                top = max(top, int(parts[0]))
    return [f"class_{i}" for i in range(top + 1)]


def _write_data_yaml(out: Path, names: List[str]) -> None:
    yaml = ["path: .", "train: train/images", "val: val/images", f"nc: {len(names)}", "names:"]
    yaml += [f"  - '{n}'" for n in names]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")


def _write_data_yaml_inplace(root: Path, names: List[str]) -> None:
    """data.yaml fuer einen vorhandenen YOLO-Ordner (images/<split> oder <split>/images)."""
    def rel(split: str) -> Optional[str]:
        for cand in (f"images/{split}", f"{split}/images", "images" if split == "train" else ""):
            if cand and (root / cand).is_dir():
                return cand
        return None
    train = rel("train") or rel("") or "images"
    val = rel("val") or rel("valid") or train
    yaml = ["path: .", f"train: {train}", f"val: {val}", f"nc: {len(names)}", "names:"] + [f"  - '{n}'" for n in names]
    (root / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")


def coco_to_yolo(root: Path, max_rows: int = 0) -> Optional[Path]:
    """COCO-JSON (images/annotations/categories) nach YOLO. Split aus dem Dateinamen."""
    jsons = []
    for f in root.rglob("*.json"):
        if f.stat().st_size > 2_000_000_000:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and {"images", "annotations", "categories"} <= set(data):
            jsons.append((f, data))
    if not jsons:
        return None
    by_name = {p.name: p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    cats = sorted({c["id"]: c["name"] for _, d in jsons for c in d["categories"]}.items())
    index = {cid: i for i, (cid, _) in enumerate(cats)}
    out = root.parent / (root.name + "_yolo")
    written = 0
    for f, data in jsons:
        low = f.as_posix().lower()
        split = "val" if any(k in low for k in ("val", "valid", "test")) else "train"
        anns: Dict[int, List[dict]] = {}
        for a in data["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        for im in data["images"]:
            if max_rows and written >= max_rows:
                break
            src = by_name.get(Path(im["file_name"]).name)
            if src is None:
                continue
            w, h = float(im.get("width") or 0), float(im.get("height") or 0)
            if not (w and h):
                from PIL import Image
                with Image.open(src) as pi:
                    w, h = pi.size
            lines = []
            for a in anns.get(im["id"], []):
                yb = _bbox_to_yolo(a.get("bbox") or [], w, h, "coco")
                if yb and a.get("category_id") in index:
                    lines.append(f"{index[a['category_id']]} {yb[0]:.6f} {yb[1]:.6f} {yb[2]:.6f} {yb[3]:.6f}")
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / split / "images" / src.name)
            (out / split / "labels" / f"{src.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            written += 1
    if not written:
        return None
    if not (out / "val" / "images").is_dir():
        _move_share_to_val(out, 0.15)
    _write_data_yaml(out, [n for _, n in cats])
    return out


def normalize(root: Path, target: Path, req: Dict[str, Any]) -> Tuple[Path, Optional[Dict[str, Any]]]:
    """Quelle in eine Form bringen, die FrameTrain trainieren kann – nur wo noetig.

    Detektion: Parquet mit Bildern -> YOLO, COCO-JSON -> YOLO, getrennte
    Bilder/Labels -> gepaart. Alles andere (Text, Tabellen, Audio, Klassenordner,
    Parquet fuer Klassifikation/ASR) lesen die Trainings-Plugins direkt.
    """
    if req.get("convert", "auto") == "none":
        return root, None
    rep = inspect(root)
    names = [str(n) for n in (req.get("class_names") or []) if str(n).strip()]
    if names and not rep["classes"] and (rep["yolo_labels"] or rep["coco_json"] == 0):
        # Klassenliste vom Aufrufer (z. B. aus der Dataset-Beschreibung), wenn die Quelle keine mitbringt
        (root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        if rep["yolo_labels"] and not list(root.glob("*.yaml")):
            _write_data_yaml_inplace(root, names)
        rep = inspect(root)
    detection = bool(req.get("yolo"))
    max_rows = int(req.get("max_rows") or 0)
    if detection and rep["images"] == 0 and rep["parquet_files"] > 0:
        emit("status", message="Converting parquet images and boxes to YOLO")
        out = target / "yolo"
        conv = parquet_to_yolo(root, out, max_rows, req.get("bbox_format") or "coco")
        return out, {"kind": "parquet_to_yolo", **conv}
    if rep["images"] and rep["coco_json"] and not rep["yolo_labels"]:
        emit("status", message="Converting COCO annotations to YOLO")
        out = coco_to_yolo(root, max_rows)
        if out:
            r2 = inspect(out)
            return out, {"kind": "coco_to_yolo", "images": r2["images"], "classes": r2["classes"]}
    if rep["images"]:
        paired = pair_images_and_labels(root)
        if paired:
            r2 = inspect(paired)
            return paired, {"kind": "paired_images_labels", "images": r2["images"], "classes": r2["classes"]}
    return root, None


# ── Einstieg ────────────────────────────────────────────────────────────────

def run(req: Dict[str, Any]) -> Dict[str, Any]:
    kind = req.get("kind")
    if kind == "inspect":
        return inspect(Path(req["path"]))
    if kind == "hf_files":
        return hf_files(req)
    target = Path(req["target"])
    target.mkdir(parents=True, exist_ok=True)
    if kind == "hf":
        emit("status", message=f"Listing {req['repo_id']}")
        root = hf_download(req, target)
    elif kind == "url":
        emit("status", message="Downloading archive")
        root = url_download(req, target)
    else:
        raise FetchError(f"Unknown kind: {kind}")
    root, conversion = normalize(root, target, req)
    report = inspect(root)
    return {"root": str(root), "report": report, "conversion": conversion}


def main(argv: List[str]) -> int:
    try:
        req = json.loads(argv[1]) if len(argv) > 1 else json.load(sys.stdin)
        emit("result", **run(req))
        return 0
    except FetchError as e:
        emit("error", message=str(e))
        return 2
    except Exception as e:  # noqa: BLE001 – lesbare Meldung statt Traceback
        emit("error", message=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
