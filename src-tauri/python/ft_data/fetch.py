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


def emit(kind: str, **data: Any) -> None:
    print(json.dumps({"type": kind, **data}, ensure_ascii=False), flush=True)


class FetchError(Exception):
    pass


# ── Pruefen ─────────────────────────────────────────────────────────────────

def inspect(path: Path) -> Dict[str, Any]:
    """Was liegt im Ordner? Bilder, YOLO-Labels, VOC, COCO, Parquet, Klassen, Splits."""
    path = Path(path)
    images = yolo = voc = coco = parquet = csv = 0
    splits: Dict[str, int] = {}
    yaml_files: List[str] = []
    for f in path.rglob("*"):
        if not f.is_file() or f.name.startswith(".") or ".frametrain_media" in f.parts:
            continue
        ext = f.suffix.lower()
        rel_parts = [p.lower() for p in f.relative_to(path).parts]
        if ext in IMAGE_EXTS:
            images += 1
            for s in ("train", "val", "valid", "validation", "test"):
                if s in rel_parts:
                    key = "val" if s in ("valid", "validation") else s
                    splits[key] = splits.get(key, 0) + 1
                    break
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
    if images and yolo:
        fmt = "yolo"
    elif images and voc:
        fmt = "voc"
    elif images and coco:
        fmt = "coco"
    elif images:
        fmt = "images"
    elif parquet:
        fmt = "parquet"
    elif csv:
        fmt = "table"
    else:
        fmt = "unknown"
    problems: List[str] = []
    if images == 0:
        problems.append("No image files found. Image and detection models (YOLO, ResNet, ...) cannot train on this dataset.")
    if images and fmt == "images":
        problems.append("Images found but no labels (YOLO .txt, Pascal VOC .xml or COCO .json). Usable only for class folders, not for object detection.")
    if fmt == "yolo" and yolo < images * 0.5:
        problems.append(f"Only {yolo} label files for {images} images. Many images are unlabeled.")
    return {
        "format": fmt, "images": images, "yolo_labels": yolo, "voc_xml": voc, "coco_json": coco,
        "parquet_files": parquet, "table_files": csv, "splits": splits, "classes": classes,
        "yaml_files": yaml_files[:5], "problems": problems,
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

def url_download(req: Dict[str, Any], target: Path) -> Path:
    import requests
    url = str(req["url"])
    if not url.lower().startswith("https://"):
        raise FetchError("Only https:// URLs are allowed.")
    max_mb = min(float(req.get("max_mb") or DEFAULT_MAX_MB), HARD_MAX_MB)
    target.mkdir(parents=True, exist_ok=True)
    name = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] or "download"
    dest = target / name
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "FrameTrain-Desktop"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        if total and total > max_mb * 1e6:
            raise FetchError(f"File is {total / 1e6:.0f} MB, above the limit of {max_mb:.0f} MB.")
        done = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if done > max_mb * 1e6:
                    raise FetchError(f"Download exceeded the limit of {max_mb:.0f} MB.")
                emit("progress", downloaded_bytes=done, total_bytes=total, message=name)
    return extract_if_archive(dest, target / "extracted")


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
    conversion = None
    if req.get("convert", "auto") != "none" and needs_yolo_conversion(root) and req.get("yolo"):
        emit("status", message="Converting parquet images and boxes to YOLO")
        yolo_dir = target / "yolo"
        conversion = parquet_to_yolo(root, yolo_dir, int(req.get("max_rows") or 0), req.get("bbox_format") or "coco")
        root = yolo_dir
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
