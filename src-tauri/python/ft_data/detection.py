"""Objekterkennung: Pascal-VOC-Annotationen fuer Ultralytics nutzbar machen.

Pascal VOC war fuer YOLO als unterstuetzt eingetragen, Ultralytics liest aber
nur .txt-Labels (klasse cx cy b h, normalisiert). Ein VOC-Dataset lief deshalb
in "Keine Labels zum Dataset gefunden". Hier werden die XML-Dateien einmalig in
YOLO-Labels umgerechnet — an genau den Ort, an dem Ultralytics sie sucht.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# Platzhalter aus dem Import (Rust generate_dataset_yaml ohne classes.txt).
PLACEHOLDER_NAMES = {"KlasseA", "KlasseB"}


def label_path_for(image: Path) -> Path:
    """Wie Ultralytics img2label_paths: letztes /images/ -> /labels/, Endung .txt."""
    s = str(image)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa in s:
        s = sb.join(s.rsplit(sa, 1))
    return Path(s.rsplit(".", 1)[0] + ".txt")


def _parse_voc(xml_path: Path) -> Tuple[Optional[str], Optional[Tuple[float, float]], List[Tuple[str, float, float, float, float]]]:
    root = ET.parse(xml_path).getroot()
    filename = (root.findtext("filename") or "").strip() or None
    size = root.find("size")
    wh = None
    if size is not None:
        try:
            w, h = float(size.findtext("width") or 0), float(size.findtext("height") or 0)
            wh = (w, h) if w > 0 and h > 0 else None
        except ValueError:
            wh = None
    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            continue
        try:
            coords = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            continue
        objects.append((name, *coords))
    return filename, wh, objects


def _read_yaml_names(yaml_path: Path) -> List[str]:
    try:
        lines = yaml_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    names: List[str] = []
    in_names = False
    for line in lines:
        stripped = line.split(" #")[0].rstrip()
        if stripped.strip().startswith("#") or not stripped.strip():
            continue
        if stripped.startswith("names:"):
            rest = stripped[len("names:"):].strip()
            if rest.startswith("[") and rest.endswith("]"):
                return [n.strip().strip("'\"") for n in rest[1:-1].split(",") if n.strip()]
            in_names = True
            continue
        if in_names:
            if not line.startswith((" ", "\t")):
                break
            item = stripped.strip()
            if item.startswith("- "):
                names.append(item[2:].strip().strip("'\""))
            elif ":" in item:
                names.append(item.split(":", 1)[1].strip().strip("'\""))
    return names


def _write_yaml_names(yaml_path: Path, names: List[str]) -> None:
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    skipping = False
    for line in lines:
        if line.startswith("nc:"):
            continue
        if line.startswith("names:"):
            skipping = True
            continue
        if skipping and (line.startswith((" ", "\t")) or not line.strip()):
            continue
        skipping = False
        out.append(line)
    while out and not out[-1].strip():
        out.pop()
    out.append("")
    out.append(f"nc: {len(names)}")
    out.append("names:")
    out.extend("  - '" + n.replace("'", "\\'") + "'" for n in names)
    yaml_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def convert_voc_to_yolo(root: Path, yaml_path: Optional[Path] = None, status=None) -> int:
    """Rechnet VOC-XML in YOLO-Labels um. Liefert die Anzahl geschriebener Labeldateien.

    Vorhandene .txt-Labels werden nie ueberschrieben. Klassennamen aus der yaml
    bleiben erhalten; stehen dort nur Platzhalter, kommen sie aus den XML-Dateien.
    """
    root = Path(root)
    xmls = [p for p in root.rglob("*.xml") if ".frametrain_media" not in p.parts]
    if not xmls:
        return 0

    images: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and ".frametrain_media" not in p.parts:
            images.setdefault(p.stem, []).append(p)

    parsed = []
    all_names = set()
    for x in xmls:
        try:
            filename, wh, objects = _parse_voc(x)
        except ET.ParseError:
            continue
        stem = Path(filename).stem if filename else x.stem
        candidates = images.get(stem) or images.get(x.stem) or []
        if not candidates:
            continue
        # Bei gleichen Namen in mehreren Splits: das Bild im selben Split wie die XML.
        split_hint = next((part for part in x.parts if part.lower() in ("train", "val", "valid", "validation", "test")), None)
        image = next((c for c in candidates if split_hint and split_hint in c.parts), candidates[0])
        parsed.append((image, wh, objects))
        all_names.update(o[0] for o in objects)

    yaml_names = _read_yaml_names(yaml_path) if yaml_path else []
    usable = [n for n in yaml_names if n not in PLACEHOLDER_NAMES]
    if usable and all_names.issubset(set(usable)):
        names = usable
    else:
        names = usable + sorted(all_names - set(usable))
    index = {n: i for i, n in enumerate(names)}

    written = 0
    for image, wh, objects in parsed:
        target = label_path_for(image)
        if target.exists():
            continue
        if wh is None:
            try:
                from PIL import Image
                with Image.open(image) as im:
                    wh = (float(im.width), float(im.height))
            except Exception:
                continue
        w, h = wh
        rows = []
        for name, x0, y0, x1, y1 in objects:
            x0, x1 = sorted((max(0.0, x0), min(w, x1)))
            y0, y1 = sorted((max(0.0, y0), min(h, y1)))
            if x1 <= x0 or y1 <= y0:
                continue
            rows.append(f"{index[name]} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        written += 1

    if yaml_path and names and names != yaml_names:
        _write_yaml_names(yaml_path, names)
    if status and written:
        status(f"Pascal VOC: {written} XML-Annotationen in YOLO-Labels umgerechnet ({len(names)} Klassen).")
    return written


def fill_placeholder_names(root: Path, yaml_path: Path, status=None) -> bool:
    """Ersetzt Platzhalter-Klassen ('KlasseA', 'KlasseB', nc: 0) durch echte Anzahl.

    Ohne classes.txt schreibt der Import Platzhalter in die yaml. Ultralytics
    bricht dann ab oder trainiert mit falscher Klassenanzahl. Die Anzahl kommt aus
    den hoechsten Klassen-IDs der .txt-Labels; die Namen lauten class_0, class_1, …
    und koennen im Tab 'dataset.yaml' umbenannt werden.
    """
    names = _read_yaml_names(yaml_path)
    real = [n for n in names if n not in PLACEHOLDER_NAMES]
    if real and len(real) == len(names):
        return False
    max_id = -1
    for txt in Path(root).rglob("*.txt"):
        if ".frametrain_media" in txt.parts or "labels" not in {p.lower() for p in txt.parts} and txt.parent.name.lower() not in ("train", "val", "test"):
            continue
        try:
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5 and re.fullmatch(r"\d+", parts[0]):
                    max_id = max(max_id, int(parts[0]))
        except (OSError, UnicodeDecodeError):
            continue
    if max_id < 0:
        return False
    new_names = real[: max_id + 1] + [f"class_{i}" for i in range(len(real[: max_id + 1]), max_id + 1)]
    _write_yaml_names(yaml_path, new_names)
    if status:
        status(f"dataset.yaml hatte Platzhalter-Klassen — {len(new_names)} Klassen aus den Labels eingetragen.")
    return True
