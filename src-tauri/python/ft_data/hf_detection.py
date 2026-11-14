"""Gemeinsame Teile fuer Train- und Test-Plugin der HF-Detektoren
(train_engine/plugins/hf_object_detection, test_engine/plugins/hf_object_detection):
Modellpruefung, Bildgroesse, Geraet und Vorhersage — an beiden Stellen gleich.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# model_type aus config.json -> lesbarer Name. Muss zu manifest.json passen.
SUPPORTED_MODEL_TYPES = {
    "detr": "DETR", "conditional_detr": "Conditional DETR", "deformable_detr": "Deformable DETR",
    "rt_detr": "RT-DETR", "rt_detr_v2": "RT-DETRv2", "yolos": "YOLOS", "dab_detr": "DAB-DETR",
    "d_fine": "D-FINE", "table-transformer": "Table Transformer",
}


def read_model_type(model_dir: Path) -> str:
    try:
        return str(json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8")).get("model_type", ""))
    except (OSError, ValueError):
        return ""


def check_model_dir(model_dir: Path) -> Optional[str]:
    """Fehlertext, wenn der Ordner kein unterstuetzter HF-Detektor ist."""
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").is_file():
        pts = sorted(p.name for p in model_dir.glob("*.pt"))
        hint = (f" Gefunden: {', '.join(pts[:3])} — das sind Ultralytics-Gewichte, dafuer ist das "
                "YOLO-Plugin zustaendig (auch fuer rtdetr-l.pt).") if pts else ""
        return f"In {model_dir} fehlt config.json; ein HuggingFace-Detektor wird erwartet.{hint}"
    mt = read_model_type(model_dir)
    if mt not in SUPPORTED_MODEL_TYPES:
        return (f"model_type '{mt or 'unbekannt'}' ist kein unterstuetzter Detektor. Unterstuetzt: "
                + ", ".join(SUPPORTED_MODEL_TYPES.values()) + ".")
    return None


def configure_processor_size(processor, imgsz: int) -> None:
    """Bildgroesse setzen: feste Hoehe/Breite (RT-DETR) oder kuerzeste/laengste Kante (DETR, YOLOS)."""
    if not imgsz:
        return
    size = getattr(processor, "size", None)
    imgsz = int(imgsz)
    if isinstance(size, dict) and "height" in size and "width" in size:
        processor.size = {"height": imgsz, "width": imgsz}
    elif isinstance(size, dict):
        processor.size = {"shortest_edge": imgsz, "longest_edge": int(round(imgsz * 1333 / 800))}


def pick_device(requested: str = "") -> str:
    import torch
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(path: str):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def predict(model, processor, images: List[Any], device: str, threshold: float = 0.0
            ) -> List[List[Dict[str, Any]]]:
    """Boxen je Bild: [{"label": id, "score", "box": [x1,y1,x2,y2] in Pixeln}]."""
    import torch
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    with torch.no_grad():
        outputs = model(**inputs)
    sizes = torch.tensor([[im.height, im.width] for im in images])
    try:
        results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=sizes)
    except TypeError:
        results = processor.post_process_object_detection(outputs, threshold, sizes)
    out = []
    for r in results:
        dets = []
        for score, label, box in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
            dets.append({"label": int(label), "score": float(score), "box": [float(v) for v in box]})
        out.append(dets)
    return out


def top_k(dets: List[Dict[str, Any]], k: int = 100) -> List[Dict[str, Any]]:
    return sorted(dets, key=lambda d: -d["score"])[:k]
