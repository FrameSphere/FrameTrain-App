"""Semantische Segmentierung: Bild + Maske, Klassenliste, Farbtabellen, mIoU.

Wird von beiden Engines genutzt (train_engine/plugins/semantic_segmentation
und test_engine/plugins/semantic_segmentation), damit Training und Test die
Masken exakt gleich lesen.

Unterstuetzte Layouts (Ordnernamen ohne Gross-/Kleinschreibung):
    <root>/images/*.jpg + <root>/masks/*.png                 ungeteilt
    <root>/train/images + <root>/train/masks [+ val/ test/]  geteilt (so teilt die App auf)
    <root>/images/train + <root>/masks/train [+ val/ test/]  geteilt (ADE20K-Stil)
Bildordner: images, image, imgs, img, jpegimages.
Maskenordner: masks, mask, labels, label, annotations, annotation,
segmentation, segmentations, segmentationclass, segmentationclasspng, gt.
Bild und Maske gehoeren ueber den Dateinamen zusammen; Endungen wie
"_mask", "_seg", "_label", "_gt" am Maskennamen werden ignoriert.

Maskenformate:
  * Klassen-IDs: Graustufen-PNG (L, I, I;16) oder Paletten-PNG (P, wie
    Pascal VOC) — der Pixelwert ist die Klassen-ID. 255 = ignorieren.
  * Farbmasken (RGB): jede Farbe ist eine Klasse. Die Zuordnung kommt aus
    einer Farbtabelle (classes.json, labelmap.txt, classes.txt mit RGB) oder
    wird aus den Masken abgeleitet (Schwarz zuerst, dann sortiert).
Klassenliste: classes.json / id2label.json / label_map.json / labelmap.txt /
classes.txt / class_names.txt / labels.txt / obj.names im Dataset-Ordner,
sonst aus den Masken (class_0, class_1, ...).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff", ".gif"}
IMAGE_DIRS = ("images", "image", "imgs", "img", "jpegimages")
MASK_DIRS = ("masks", "mask", "labels", "label", "annotations", "annotation",
             "segmentation", "segmentations", "segmentationclass",
             "segmentationclasspng", "gt")
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation", "dev"),
    "test": ("test", "testing"),
}
CLASS_FILES = ("classes.json", "id2label.json", "label_map.json", "labelmap.json",
               "labelmap.txt", "classes.txt", "class_names.txt", "labels.txt", "obj.names")
MASK_SUFFIXES = ("_segmentation", "_labelids", "_labels", "_label", "_mask", "_seg", "_gt",
                 "-mask", "-seg", "-label")
IGNORE_INDEX = 255

Pair = Tuple[Path, Path]
Color = Tuple[int, int, int]


# ── Layout ───────────────────────────────────────────────────────────────────

def _dirs(d: Path) -> List[Path]:
    try:
        return sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith((".", "__")))
    except OSError:
        return []


def _find_dir(base: Path, names: Sequence[str]) -> Optional[Path]:
    by_name = {p.name.lower(): p for p in _dirs(base)}
    for n in names:
        if n in by_name:
            return by_name[n]
    return None


def _files(d: Path, exts: set) -> List[Path]:
    return sorted(f for f in d.rglob("*")
                  if f.is_file() and f.suffix.lower() in exts and not f.name.startswith("."))


def mask_key(stem: str) -> str:
    s = stem.lower()
    for suf in MASK_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)]
    return s


def pair_files(img_dir: Path, mask_dir: Path) -> Tuple[List[Pair], int]:
    """(Paare, Bilder ohne Maske). Paarung ueber den Dateinamen."""
    masks: Dict[str, Path] = {}
    for m in _files(mask_dir, MASK_EXTS):
        masks.setdefault(mask_key(m.stem), m)
    pairs: List[Pair] = []
    orphans = 0
    for img in _files(img_dir, IMAGE_EXTS):
        m = masks.get(img.stem.lower()) or masks.get(mask_key(img.stem))
        if m is None or m.resolve() == img.resolve():
            orphans += 1
            continue
        pairs.append((img, m))
    return pairs, orphans


def _pair_dirs(base: Path) -> Optional[Tuple[Path, Path]]:
    img = _find_dir(base, IMAGE_DIRS)
    msk = _find_dir(base, MASK_DIRS)
    if img is None or msk is None or img == msk:
        return None
    return img, msk


def _split_of(name: str) -> Optional[str]:
    n = name.lower()
    for split, aliases in SPLIT_ALIASES.items():
        if n in aliases:
            return split
    return None


@dataclass
class SegLayout:
    train: List[Pair] = field(default_factory=list)
    val: List[Pair] = field(default_factory=list)
    test: List[Pair] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def all_masks(self) -> List[Path]:
        return [m for _, m in self.train + self.val + self.test]


def find_splits(root: Path) -> Tuple[Dict[str, List[Pair]], int]:
    """Alle Splits mit Paaren: {"train": [...], ...}, Anzahl Bilder ohne Maske."""
    root = Path(root)
    out: Dict[str, List[Pair]] = {}
    orphans = 0
    # A: <split>/images + <split>/masks (so legt der App-Split an)
    for d in _dirs(root):
        split = _split_of(d.name)
        if not split or split in out:
            continue
        pd = _pair_dirs(d)
        if pd:
            pairs, o = pair_files(*pd)
            orphans += o
            if pairs:
                out[split] = pairs
    if out:
        return out, orphans
    pd = _pair_dirs(root)
    if pd is None:
        return {}, 0
    img_dir, mask_dir = pd
    # B: images/<split> + masks/<split>
    for d in _dirs(img_dir):
        split = _split_of(d.name)
        if not split or split in out:
            continue
        md = _find_dir(mask_dir, SPLIT_ALIASES[split])
        if md is not None:
            pairs, o = pair_files(d, md)
            orphans += o
            if pairs:
                out[split] = pairs
    if out:
        return out, orphans
    # C: images/ + masks/ ungeteilt
    pairs, orphans = pair_files(img_dir, mask_dir)
    if pairs:
        out["train"] = pairs
    return out, orphans


def resolve_layout(root: Path, seed: int = 42, val_fraction: float = 0.2) -> SegLayout:
    """Train/Val/Test fuer das Training. test/ bleibt fuer den Test unberuehrt."""
    root = Path(root)
    splits, orphans = find_splits(root)
    if not splits:
        raise ValueError(
            f"Keine Bild/Masken-Paare in {root} gefunden. Erwartet: images/ + masks/ "
            "(gleiche Dateinamen, Masken als PNG), optional in train/ val/ test/.")
    lay = SegLayout(train=list(splits.get("train", [])), val=list(splits.get("val", [])),
                    test=list(splits.get("test", [])))
    if orphans:
        lay.notes.append(f"{orphans} Bild(er) ohne passende Maske werden uebersprungen.")
    if not lay.train:
        src = "val" if lay.val else "test"
        lay.train = list(getattr(lay, src))
        setattr(lay, src, [])
        lay.notes.append(f"Kein train-Split gefunden - {src}/ wird zum Trainieren genutzt.")
    if not lay.val and len(lay.train) >= 2:
        items = list(lay.train)
        random.Random(seed).shuffle(items)
        n_val = max(1, int(round(len(items) * val_fraction)))
        n_val = min(n_val, len(items) - 1)
        lay.val, lay.train = sorted(items[:n_val]), sorted(items[n_val:])
        lay.notes.append(f"Kein val-Split - {n_val} von {len(items)} Paaren werden zur Validierung abgetrennt.")
    elif not lay.val:
        lay.val = list(lay.train)
        lay.notes.append("Nur ein Bild/Masken-Paar - Validierung laeuft auf dem Trainingsbild.")
    return lay


def evaluation_pairs(root: Path) -> Tuple[Optional[str], List[Pair]]:
    """Paare fuer einen Testlauf: test vor val vor train."""
    splits, _ = find_splits(Path(root))
    for split in ("test", "val", "train"):
        if splits.get(split):
            return split, splits[split]
    return None, []


# ── Klassenliste und Farbtabelle ─────────────────────────────────────────────

@dataclass
class ClassInfo:
    names: List[str]
    colors: Optional[List[Optional[Color]]] = None
    source: str = ""

    @property
    def has_colors(self) -> bool:
        return bool(self.colors) and all(c is not None for c in self.colors)


def _color(v: Any) -> Optional[Color]:
    if isinstance(v, str):
        s = v.strip().lstrip("#")
        if re.fullmatch(r"[0-9a-fA-F]{6}", s):
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        parts = re.findall(r"\d+", v)
        v = [int(p) for p in parts]
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return (int(v[0]), int(v[1]), int(v[2]))
    return None


def _from_id_map(d: Dict[Any, Any]) -> List[str]:
    ids = {int(k): str(v) for k, v in d.items()}
    n = max(ids) + 1 if ids else 0
    return [ids.get(i, f"class_{i}") for i in range(n)]


def _parse_json(data: Any) -> Optional[ClassInfo]:
    if isinstance(data, dict):
        if isinstance(data.get("id2label"), dict):
            return ClassInfo(_from_id_map(data["id2label"]))
        lst = next((data[k] for k in ("classes", "names", "labels", "categories") if isinstance(data.get(k), list)), None)
        if lst is not None:
            info = _parse_json(lst)
            pal = next((data[k] for k in ("colors", "palette", "colours") if isinstance(data.get(k), list)), None)
            if info and pal is not None:
                info.colors = [_color(c) for c in pal][: len(info.names)]
            return info
        if data and all(re.fullmatch(r"\d+", str(k)) for k in data):
            return ClassInfo(_from_id_map(data))
        return None
    if isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            return ClassInfo([str(x) for x in data])
        if all(isinstance(x, dict) for x in data):
            rows = []
            for i, x in enumerate(data):
                name = x.get("name", x.get("label", x.get("class", f"class_{i}")))
                cid = int(x.get("id", x.get("index", i)))
                rows.append((cid, str(name), _color(x.get("color", x.get("rgb", x.get("colour"))))))
            n = max(r[0] for r in rows) + 1 if rows else 0
            names = [f"class_{i}" for i in range(n)]
            colors: List[Optional[Color]] = [None] * n
            for cid, name, col in rows:
                names[cid], colors[cid] = name, col
            return ClassInfo(names, colors if any(c is not None for c in colors) else None)
    return None


_TXT_COLOR = re.compile(r"^(.+?)[\s:,;]+(\d{1,3})[\s,;]+(\d{1,3})[\s,;]+(\d{1,3})\s*$")
_TXT_ID = re.compile(r"^(\d+)[\s:,;]+(.+)$")


def _parse_txt(text: str, labelmap: bool) -> Optional[ClassInfo]:
    names: List[str] = []
    colors: List[Optional[Color]] = []
    by_id: Dict[int, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if labelmap:   # CVAT/VOC: name:r,g,b:parts:actions
            parts = line.split(":")
            names.append(parts[0].strip())
            colors.append(_color(parts[1]) if len(parts) > 1 else None)
            continue
        m = _TXT_COLOR.match(line)
        if m:
            names.append(m.group(1).strip())
            colors.append((int(m.group(2)), int(m.group(3)), int(m.group(4))))
            continue
        m = _TXT_ID.match(line)
        if m:
            by_id[int(m.group(1))] = m.group(2).strip()
            continue
        names.append(line)
        colors.append(None)
    if by_id and not names:
        return ClassInfo(_from_id_map(by_id))
    if not names:
        return None
    return ClassInfo(names, colors if any(c is not None for c in colors) else None)


def read_class_file(root: Path) -> Optional[ClassInfo]:
    root = Path(root)
    lower = {p.name.lower(): p for p in root.iterdir() if p.is_file()} if root.is_dir() else {}
    for name in CLASS_FILES:
        f = lower.get(name)
        if f is None:
            continue
        try:
            text = f.read_text(encoding="utf-8")
            info = (_parse_json(json.loads(text)) if f.suffix.lower() == ".json"
                    else _parse_txt(text, labelmap=name == "labelmap.txt"))
        except Exception:
            info = None
        if info and info.names:
            info.source = f.name
            return info
    return None


def default_palette(n: int) -> List[Color]:
    """Pascal-VOC-Farbtabelle: 0 schwarz, danach gut unterscheidbare Farben."""
    out = []
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        out.append((r, g, b))
    return out


# ── Masken lesen ─────────────────────────────────────────────────────────────

def _raw_mask(path: Path) -> Tuple[np.ndarray, str]:
    """(Array, 'index'|'color'). Paletten-PNG liefert die Indizes, nicht die Farben."""
    from PIL import Image
    with Image.open(path) as im:
        if im.mode == "P":
            return np.array(im), "index"
        if im.mode in ("L", "I", "I;16", "I;16B", "I;16L", "1", "F"):
            return np.array(im).astype(np.int64), "index"
        arr = np.array(im.convert("RGB"))
    if (arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all():
        return arr[..., 0].astype(np.int64), "index"
    return arr, "color"


def _color_keys(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.int64)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


@dataclass
class MaskDecoder:
    mode: str                       # "index" | "color"
    num_classes: int
    names: List[str]
    colors: List[Color]             # Anzeigefarben (bei "color": auch die Decodier-Tabelle)
    ignore_index: int = IGNORE_INDEX
    unknown_pixels: int = 0

    def decode(self, path: Path) -> np.ndarray:
        arr, kind = _raw_mask(path)
        if kind == "color" and self.mode == "color":
            keys = _color_keys(arr)
            lut = {(r << 16) | (g << 8) | b: i for i, (r, g, b) in enumerate(self.colors)}
            uniq, inv = np.unique(keys, return_inverse=True)
            mapped = np.array([lut.get(int(k), self.ignore_index) for k in uniq], dtype=np.int64)
            self.unknown_pixels += int(np.sum(mapped[inv] == self.ignore_index))
            return mapped[inv].reshape(keys.shape)
        if kind == "color":
            raise ValueError(f"{path.name} ist eine Farbmaske, die anderen Masken enthalten Klassen-IDs.")
        out = arr.astype(np.int64)
        out[(out >= self.num_classes) & (out != self.ignore_index)] = self.ignore_index
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"mask_mode": self.mode, "classes": self.names,
                "colors": [list(c) for c in self.colors], "ignore_index": self.ignore_index}


def build_decoder(mask_paths: Sequence[Path], info: Optional[ClassInfo] = None,
                  fallback: Optional[Dict[str, Any]] = None, scan_limit: int = 2000,
                  num_classes: Optional[int] = None) -> MaskDecoder:
    """Maskenformat erkennen und Klassen festlegen.

    info:     Klassenliste aus dem Dataset (read_class_file).
    fallback: label_mapping.json eines trainierten Modells (Test-Engine) —
              liefert Namen/Farben, wenn das Dataset selbst keine hat.
    num_classes: feste Klassenzahl (Test: die des Modells).
    """
    paths = list(mask_paths)
    if not paths:
        raise ValueError("Keine Masken gefunden.")
    step = max(1, len(paths) // scan_limit)
    sample = paths[::step][:scan_limit]
    modes = set()
    ids: set = set()
    colors: set = set()
    for p in sample:
        arr, kind = _raw_mask(p)
        modes.add(kind)
        if kind == "index":
            ids.update(int(v) for v in np.unique(arr))
        else:
            colors.update(int(k) for k in np.unique(_color_keys(arr)))
    if len(modes) > 1:
        raise ValueError("Die Masken mischen Klassen-IDs und Farbmasken - bitte ein Format verwenden.")
    mode = modes.pop()
    fb_names = list((fallback or {}).get("classes") or [])
    fb_colors = [tuple(c) for c in ((fallback or {}).get("colors") or [])]

    if mode == "index":
        if num_classes is None:
            if info and info.names:
                num_classes = len(info.names)
            elif fb_names:
                num_classes = len(fb_names)
            else:
                real = [i for i in ids if i != IGNORE_INDEX]
                num_classes = (max(real) + 1) if real else 1
        names = (info.names if info and len(info.names) == num_classes else
                 fb_names if len(fb_names) == num_classes else
                 [f"class_{i}" for i in range(num_classes)])
        if info and info.has_colors and len(info.colors) == num_classes:
            pal = [tuple(c) for c in info.colors]
        elif len(fb_colors) == num_classes:
            pal = fb_colors
        else:
            pal = default_palette(num_classes)
        return MaskDecoder("index", num_classes, list(names), pal)

    # Farbmasken
    if info and info.has_colors:
        table = [tuple(c) for c in info.colors]
        names = list(info.names)
    elif fb_colors and (fallback or {}).get("mask_mode") == "color":
        table, names = fb_colors, fb_names
    else:
        found = sorted(((k >> 16) & 255, (k >> 8) & 255, k & 255) for k in colors)
        found.sort(key=lambda c: (c != (0, 0, 0), c))
        table = found
        names = (list(info.names) if info and len(info.names) == len(found)
                 else [f"class_{i}" for i in range(len(found))])
    if num_classes is not None and len(table) != num_classes:
        raise ValueError(
            f"Die Farbtabelle hat {len(table)} Farben, das Modell kennt {num_classes} Klassen.")
    return MaskDecoder("color", len(table), names, table)


# ── Vorverarbeitung ──────────────────────────────────────────────────────────

def load_rgb(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def prepare_pair(image, mask: Optional[np.ndarray], size: Tuple[int, int],
                 mean: Sequence[float], std: Sequence[float], flip: bool = False,
                 rescale: float = 1 / 255) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Bild (bilinear) und Maske (nearest) auf size=(H, W), normalisiert, CHW."""
    from PIL import Image
    h, w = size
    img = image.resize((w, h), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) * rescale
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    x = x.transpose(2, 0, 1)
    y = None
    if mask is not None:
        m = Image.fromarray(mask.astype(np.int32), mode="I")
        y = np.asarray(m.resize((w, h), Image.NEAREST), dtype=np.int64)
    if flip:
        x = x[:, :, ::-1]
        y = y[:, ::-1] if y is not None else None
    return np.ascontiguousarray(x), (np.ascontiguousarray(y) if y is not None else None)


def processor_settings(processor, default_size: int = 512) -> Dict[str, Any]:
    """Mittelwert, Streuung, Skalierung und Groesse aus dem HF-Bild-Prozessor."""
    mean = list(getattr(processor, "image_mean", None) or [0.485, 0.456, 0.406])
    std = list(getattr(processor, "image_std", None) or [0.229, 0.224, 0.225])
    do_rescale = getattr(processor, "do_rescale", True)
    rescale = float(getattr(processor, "rescale_factor", 1 / 255) or 1 / 255) if do_rescale is not False else 1.0
    if getattr(processor, "do_normalize", True) is False:
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    size = getattr(processor, "size", None) or {}
    if hasattr(size, "items"):
        size = dict(size)
    h = w = default_size
    if isinstance(size, dict):
        if size.get("height") and size.get("width"):
            h, w = int(size["height"]), int(size["width"])
        elif size.get("shortest_edge"):
            h = w = int(size["shortest_edge"])
    elif isinstance(size, int):
        h = w = size
    crop = getattr(processor, "crop_size", None)
    if isinstance(crop, dict) and crop.get("height") and getattr(processor, "do_center_crop", False):
        h, w = int(crop["height"]), int(crop["width"])
    return {"mean": mean, "std": std, "rescale": rescale, "size": (h, w)}


# ── Kennzahlen ───────────────────────────────────────────────────────────────

def confusion(pred: np.ndarray, label: np.ndarray, num_classes: int,
              ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Konfusionsmatrix [wahr, vorhergesagt] ueber alle gueltigen Pixel."""
    pred = np.asarray(pred, dtype=np.int64).ravel()
    label = np.asarray(label, dtype=np.int64).ravel()
    keep = (label != ignore_index) & (label >= 0) & (label < num_classes)
    idx = num_classes * label[keep] + np.clip(pred[keep], 0, num_classes - 1)
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def scores(cm: np.ndarray, names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """mIoU, Pixel-Accuracy, mittlere Klassen-Accuracy, IoU je Klasse.

    Klassen, die weder im Label noch in der Vorhersage vorkommen, zaehlen nicht
    in den Mittelwert (IoU None) — sonst druecken leere Klassen das mIoU.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    gt = cm.sum(axis=1)
    pr = cm.sum(axis=0)
    union = gt + pr - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
        acc = np.where(gt > 0, tp / gt, np.nan)
    total = cm.sum()
    n = cm.shape[0]
    names = list(names) if names is not None and len(names) == n else [f"class_{i}" for i in range(n)]
    valid = ~np.isnan(iou)
    return {
        "mean_iou": float(np.mean(iou[valid])) if valid.any() else 0.0,
        "pixel_accuracy": float(tp.sum() / total) if total > 0 else 0.0,
        "mean_accuracy": float(np.nanmean(acc)) if (~np.isnan(acc)).any() else 0.0,
        "per_class_iou": {names[i]: (None if np.isnan(iou[i]) else float(iou[i])) for i in range(n)},
        "per_class_accuracy": {names[i]: (None if np.isnan(acc[i]) else float(acc[i])) for i in range(n)},
    }


# ── Darstellung ──────────────────────────────────────────────────────────────

def colorize(pred: np.ndarray, colors: Sequence[Color]) -> np.ndarray:
    pal = np.array(list(colors) + [(0, 0, 0)], dtype=np.uint8)
    idx = np.where((pred >= 0) & (pred < len(colors)), pred, len(colors))
    return pal[idx]


def save_prediction(image, pred: np.ndarray, colors: Sequence[Color], out_dir: Path,
                    stem: str, alpha: float = 0.5) -> Dict[str, str]:
    """Speichert Klassen-ID-Maske, Farbmaske und Ueberlagerung. -> Pfade."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    color = colorize(pred, colors)
    paths = {
        "mask_path": out_dir / f"{stem}_mask.png",
        "color_mask_path": out_dir / f"{stem}_color.png",
        "overlay_path": out_dir / f"{stem}_overlay.png",
    }
    Image.fromarray(pred.astype(np.uint8 if pred.max(initial=0) < 256 else np.uint16)).save(paths["mask_path"])
    Image.fromarray(color).save(paths["color_mask_path"])
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    blend = (base * (1 - alpha) + color.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(blend).save(paths["overlay_path"])
    return {k: str(v) for k, v in paths.items()}


def legend(pred: np.ndarray, names: Sequence[str], colors: Sequence[Color]) -> List[Dict[str, Any]]:
    """Klassen im Bild mit Flaechenanteil, groesste zuerst."""
    total = max(int(pred.size), 1)
    ids, counts = np.unique(pred, return_counts=True)
    rows = []
    for i, c in zip(ids.tolist(), counts.tolist()):
        if 0 <= i < len(names):
            r, g, b = colors[i]
            rows.append({"id": int(i), "label": names[i], "color": f"#{r:02x}{g:02x}{b:02x}",
                         "share": c / total})
    return sorted(rows, key=lambda r: -r["share"])


StatusFn = Optional[Callable[[str], None]]
