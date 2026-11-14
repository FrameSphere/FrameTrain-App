"""Objekterkennungs-Daten fuer Nicht-Ultralytics-Detektoren (DETR & Co.).

Liest die beiden Formate, die FrameTrain fuer Boxen kennt, in eine gemeinsame
Form:
  - YOLO: images/ + labels/*.txt (klasse cx cy w h, normiert), Klassennamen aus
    data.yaml/dataset.yaml oder classes.txt; Splits train/val/test in den
    ueblichen Layouts (train/images, images/train).
  - COCO-JSON: eine Datei je Split (instances_train.json, train/_annotations.coco.json
    von Roboflow, annotations/*.json) mit images/annotations/categories.

Ein Datensatz ist {"path", "width", "height", "boxes": [{"label": int, "box":
[x1,y1,x2,y2]}]} mit Pixelkoordinaten und fortlaufenden Klassen-IDs 0..n-1.
Daraus erzeugt to_coco_target() die Annotation, die die HuggingFace-
Bildprozessoren erwarten.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


# ── YOLO ────────────────────────────────────────────────────────────────────
def yolo_line_to_box(line: str, width: int, height: int) -> Optional[Tuple[int, List[float]]]:
    """'klasse cx cy w h' (normiert) -> (klasse, [x1,y1,x2,y2] in Pixeln).

    Zeilen mit mehr Werten (Polygone) werden zur umschliessenden Box.
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if len(vals) == 4:
        cx, cy, w, h = vals
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    else:
        xs, ys = vals[0::2], vals[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    x1, x2 = max(0.0, x1) * width, min(1.0, x2) * width
    y1, y2 = max(0.0, y1) * height, min(1.0, y2) * height
    if x2 <= x1 or y2 <= y1:
        return None
    return cls, [x1, y1, x2, y2]


def label_path_for(image: Path) -> Optional[Path]:
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            if cand.exists():
                return cand
            break
    for cand in (image.parent.parent / "labels" / (image.stem + ".txt"), image.with_suffix(".txt")):
        if cand.exists():
            return cand
    return None


def image_size(path: Path) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def _images_in(d: Path) -> List[Path]:
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                  and "labels" not in p.parts)


def yolo_records(images_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for img in _images_in(images_dir):
        w, h = image_size(img)
        boxes = []
        lab = label_path_for(img)
        if lab is not None:
            for line in lab.read_text(encoding="utf-8", errors="replace").splitlines():
                parsed = yolo_line_to_box(line, w, h)
                if parsed:
                    boxes.append({"label": parsed[0], "box": parsed[1]})
        out.append({"path": str(img), "width": w, "height": h, "boxes": boxes})
    return out


def read_yolo_names(root: Path) -> List[str]:
    for name in ("data.yaml", "dataset.yaml"):
        p = root / name
        if p.is_file():
            try:
                import yaml
                d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                names = d.get("names")
                if isinstance(names, dict):
                    return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
                if isinstance(names, list) and names:
                    return [str(n) for n in names]
            except Exception:
                pass
    for name in ("classes.txt", "obj.names", "labels.txt"):
        p = root / name
        if p.is_file():
            names = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            if names:
                return names
    return []


def yolo_split_dirs(root: Path) -> Dict[str, Path]:
    """Bildordner je Split (train/images, images/train, train/)."""
    found: Dict[str, Path] = {}
    for split, aliases in SPLIT_ALIASES.items():
        for a in aliases:
            for cand in (root / a / "images", root / "images" / a, root / a):
                if cand.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in cand.rglob("*")):
                    found[split] = cand
                    break
            if split in found:
                break
    if not found and (root / "images").is_dir():
        found["train"] = root / "images"
    return found


# ── COCO ────────────────────────────────────────────────────────────────────
def is_coco_json(path: Path) -> bool:
    try:
        if path.stat().st_size > 512 * 1024 * 1024:
            return False
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(d, dict) and isinstance(d.get("images"), list) and isinstance(d.get("annotations"), list)


def coco_records(json_path: Path, categories_order: Optional[List[int]] = None
                 ) -> Tuple[List[Dict[str, Any]], List[str], List[int]]:
    """COCO-Datei -> (Datensaetze, Klassennamen, Kategorie-IDs in Reihenfolge).

    COCO-IDs sind oft lueckenhaft (1..90); intern zaehlen sie fortlaufend.
    """
    d = json.loads(json_path.read_text(encoding="utf-8"))
    cats = sorted(d.get("categories") or [], key=lambda c: c["id"])
    order = categories_order or [c["id"] for c in cats]
    names_by_id = {c["id"]: str(c.get("name", c["id"])) for c in cats}
    index = {cid: i for i, cid in enumerate(order)}
    anns: Dict[Any, List[dict]] = {}
    for a in d.get("annotations", []):
        anns.setdefault(a.get("image_id"), []).append(a)
    base_candidates = [json_path.parent, json_path.parent / "images", json_path.parent.parent / "images",
                       json_path.parent.parent]
    records = []
    for im in d.get("images", []):
        fname = str(im.get("file_name", ""))
        path = next((b / fname for b in base_candidates if (b / fname).is_file()), None)
        if path is None:
            hits = list(json_path.parent.parent.rglob(Path(fname).name))
            path = hits[0] if hits else None
        if path is None:
            continue
        w, h = int(im.get("width") or 0), int(im.get("height") or 0)
        if not w or not h:
            w, h = image_size(path)
        boxes = []
        for a in anns.get(im.get("id"), []):
            if a.get("iscrowd"):
                continue
            cid = a.get("category_id")
            if cid not in index or not a.get("bbox"):
                continue
            x, y, bw, bh = (float(v) for v in a["bbox"][:4])
            if bw <= 0 or bh <= 0:
                continue
            boxes.append({"label": index[cid], "box": [x, y, x + bw, y + bh]})
        records.append({"path": str(path), "width": w, "height": h, "boxes": boxes})
    names = [names_by_id.get(cid, str(cid)) for cid in order]
    return records, names, order


def coco_split_files(root: Path) -> Dict[str, Path]:
    """COCO-Dateien je Split, am Datei- oder Ordnernamen erkannt."""
    files = [p for p in sorted(root.rglob("*.json")) if p.stat().st_size > 0 and is_coco_json(p)]
    out: Dict[str, Path] = {}
    for p in files:
        hay = "/".join(x.lower() for x in p.relative_to(root).parts)
        for split, aliases in SPLIT_ALIASES.items():
            if split not in out and any(a in hay for a in aliases):
                out[split] = p
    if not out and files:
        out["train"] = files[0]
    return out


# ── Gemeinsam ───────────────────────────────────────────────────────────────
def load_detection_dataset(root: Path, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """{"format", "classes", "splits": {split: [Datensatz]}} fuer YOLO oder COCO.

    class_names gibt beim Testen die Klassen des Modells vor: COCO-Kategorien
    werden ueber den Namen darauf abgebildet, YOLO-IDs bleiben, wie sie sind.
    """
    root = Path(root)
    coco = coco_split_files(root)
    yolo = yolo_split_dirs(root)
    has_yolo_labels = any((root / s / "labels").is_dir() for s in ("train", "val", "valid", "test")) \
        or (root / "labels").is_dir()
    if coco and not has_yolo_labels:
        first = coco.get("train") or next(iter(coco.values()))
        _, names, order = coco_records(first)
        if class_names:
            by_name = {n: cid for n, cid in zip(names, order)}
            order = [by_name.get(n, -1 - i) for i, n in enumerate(class_names)]
            names = list(class_names)
        splits = {s: coco_records(p, order)[0] for s, p in coco.items()}
        return {"format": "coco_json", "classes": names, "splits": splits}
    if yolo:
        splits = {s: yolo_records(d) for s, d in yolo.items()}
        names = list(class_names or read_yolo_names(root))
        top = max((b["label"] for recs in splits.values() for r in recs for b in r["boxes"]), default=-1)
        if len(names) <= top:
            names += [f"class_{i}" for i in range(len(names), top + 1)]
        return {"format": "yolo_bbox", "classes": names, "splits": splits}
    raise ValueError(
        f"In '{root}' wurden weder YOLO-Labels (images/ + labels/*.txt) noch eine COCO-JSON-Datei "
        "(images, annotations, categories) gefunden.")


def to_coco_target(record: Dict[str, Any], image_id: int) -> Dict[str, Any]:
    """Annotation im COCO-Format, wie AutoImageProcessor(annotations=...) sie liest."""
    anns = []
    for b in record["boxes"]:
        x1, y1, x2, y2 = b["box"]
        w, h = x2 - x1, y2 - y1
        anns.append({"image_id": image_id, "category_id": int(b["label"]), "bbox": [x1, y1, w, h],
                     "area": w * h, "iscrowd": 0})
    return {"image_id": image_id, "annotations": anns}


def hflip_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Boxen fuer ein horizontal gespiegeltes Bild."""
    w = record["width"]
    flipped = [{"label": b["label"], "box": [w - b["box"][2], b["box"][1], w - b["box"][0], b["box"][3]]}
               for b in record["boxes"]]
    return {**record, "boxes": flipped}
