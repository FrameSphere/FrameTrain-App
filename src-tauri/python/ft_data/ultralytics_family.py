"""Welche Ultralytics-Modellfamilie steckt in einer Gewichtsdatei?

Ultralytics laedt nicht alles ueber YOLO(): RT-DETR braucht die Klasse RTDETR,
YOLO-World die Klasse YOLOWorld, YOLOE die Klasse YOLOE. YOLO() schaltet zwar
teilweise selbst um (RT-DETR am Kopf, World/YOLOE am Dateinamen), aber ein
trainiertes "model.pt" verraet seinen Namen nicht mehr. Deshalb merkt sich das
Train-Plugin die Familie in model.json, und Train- wie Test-Plugin laden ueber
load_ultralytics().

Stand Ultralytics 8.4.80 (gemessen mit je einer Epoche auf coco128-Ausschnitt):
  - YOLOv5u/v8/v9/v10, YOLO11, YOLO12, YOLO26: trainieren und testen.
  - RT-DETR (rtdetr-l/-x): trainiert und testet (Klasse RTDETR).
  - YOLO-World: testen geht, Training braucht das Paket "clip" (CLIP-Textmodell).
  - YOLOE: Training ist nicht eingebunden (eigene Trainer, Textmodell).
  - "-sem" (semantische Segmentierung, YOLO26): nicht Teil dieses Plugins.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Optional, Tuple

FAMILIES = ("rtdetr", "yolo_world", "yoloe", "yolov10", "yolo12", "yolo26", "yolo")


def family_from_name(name: str) -> Optional[str]:
    """Familie aus dem Dateinamen; None, wenn der Name nichts verraet (model.pt)."""
    stem = Path(str(name or "")).stem.lower()
    if not stem:
        return None
    if stem.startswith("rtdetr") or stem.startswith("rt-detr"):
        return "rtdetr"
    if "-world" in stem or stem.startswith("yolo-world") or stem.startswith("yolo_world"):
        return "yolo_world"
    if stem.startswith("yoloe"):
        return "yoloe"
    if re.match(r"yolov10", stem):
        return "yolov10"
    if re.match(r"yolo12", stem):
        return "yolo12"
    if re.match(r"yolo26", stem):
        return "yolo26"
    if stem.startswith("yolo"):
        return "yolo"
    return None


def is_semantic_weights(name: str) -> bool:
    return Path(str(name or "")).stem.lower().endswith("-sem")


def family_from_meta(weights: Path) -> Optional[str]:
    """Familie aus der model.json, die das Train-Plugin neben model.pt schreibt."""
    meta = Path(weights).with_suffix(".json")
    if not meta.is_file():
        meta = Path(weights).parent / "model.json"
    try:
        d = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    fam = d.get("family")
    if fam in FAMILIES:
        return fam
    return family_from_name(str(d.get("base_model") or ""))


def family_of_model(model) -> str:
    """Familie eines geladenen Ultralytics-Modells (an der Klasse bzw. dem Kopf)."""
    cls = type(model).__name__
    if cls == "RTDETR":
        return "rtdetr"
    if cls == "YOLOWorld":
        return "yolo_world"
    if cls == "YOLOE":
        return "yoloe"
    try:
        head = model.model.model[-1]._get_name()
        if "RTDETR" in head:
            return "rtdetr"
        if "WorldDetect" in head:
            return "yolo_world"
        if "v10Detect" in head:
            return "yolov10"
    except Exception:
        pass
    return "yolo"


def resolve_family(weights: str) -> str:
    return family_from_meta(Path(weights)) or family_from_name(weights) or "yolo"


def load_ultralytics(weights: str, family: Optional[str] = None):
    """Laedt die Gewichte mit der passenden Ultralytics-Klasse."""
    import ultralytics
    fam = family or resolve_family(weights)
    if fam == "rtdetr":
        return ultralytics.RTDETR(weights)
    if fam == "yolo_world" and hasattr(ultralytics, "YOLOWorld"):
        return ultralytics.YOLOWorld(weights)
    if fam == "yoloe" and hasattr(ultralytics, "YOLOE"):
        return ultralytics.YOLOE(weights)
    return ultralytics.YOLO(weights)


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def training_blocker(weights: str, family: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """(Titel, Erklaerung), wenn diese Gewichte hier nicht trainierbar sind — sonst None.

    Ohne diese Pruefung endeten solche Laeufe mitten im Training mit
    "No module named 'clip'" oder einem Tensor-Formfehler, oder Ultralytics
    versuchte, Pakete per pip nachzuinstallieren.
    """
    import sys
    name = Path(str(weights)).name
    if is_semantic_weights(name):
        return ("Semantische Segmentierung wird hier nicht trainiert",
                f"{name} ist ein Modell fuer semantische Segmentierung (Ultralytics-Aufgabe 'semantic'). "
                "Das YOLO-Plugin trainiert Erkennung, Instanz-Segmentierung (-seg), Keypoints (-pose), "
                "gedrehte Boxen (-obb) und Klassifikation (-cls). Waehle z.B. yolo11n-seg.pt.")
    fam = family or resolve_family(weights)
    if fam == "yolo_world" and not _has_module("clip"):
        return ("YOLO-World braucht zum Training das Paket 'clip'",
                f"{name} ist ein YOLO-World-Modell. Zum Testen reicht Ultralytics, zum Training "
                "berechnet Ultralytics die Klassen-Texte mit dem CLIP-Textmodell — das Paket fehlt.\n"
                "Installieren mit:\n"
                f"  {sys.executable} -m pip install git+https://github.com/ultralytics/CLIP.git\n"
                "Oder fuer ein festes Klassenset ein normales Erkennungsmodell trainieren "
                "(yolo11n.pt, yolo26n.pt oder rtdetr-l.pt).")
    if fam == "yoloe":
        return ("YOLOE-Training ist nicht eingebunden",
                f"{name} ist ein YOLOE-Modell (offene Klassen per Text-/Bildvorgabe). Ultralytics trainiert "
                "es mit eigenen Trainern und einem Textmodell (mobileclip); das unterstuetzt FrameTrain "
                "nicht. Fuer ein eigenes Klassenset yolo11n-seg.pt (Masken) oder yolo11n.pt (Boxen) waehlen.")
    return None


def default_hyperparams(family: str) -> Tuple[str, float]:
    """(optimizer, lr0) fuer eine Familie, wenn der Nutzer nichts geaendert hat.

    RT-DETR ist ein Transformer: mit SGD und lr0 0.01 (YOLO-Standard) divergiert
    das Fine-Tuning; Ultralytics selbst nimmt dort AdamW mit 1e-4.
    """
    if family == "rtdetr":
        return "AdamW", 0.0001
    return "SGD", 0.01
