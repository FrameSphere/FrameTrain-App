"""Vorhersage fuer den Inferenz-Tab: Zahlen, Text, Dateien und Erzeugen."""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_CACHE: Dict[str, Tuple[float, Any]] = {}


def _load(model_dir: str) -> Any:
    canvas = Path(__file__).resolve().parent.parent / "train_engine" / "plugins" / "canvas"
    if str(canvas) not in sys.path:
        sys.path.insert(0, str(canvas))
    from loader import load_canvas_model
    stamp = os.path.getmtime(Path(model_dir) / "model.pt")
    cached = _CACHE.get(model_dir)
    if cached is None or cached[0] != stamp:
        _CACHE[model_dir] = (stamp, load_canvas_model(model_dir))
    return _CACHE[model_dir][1]


def _png(image: Any) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _tensor_images(batch: torch.Tensor) -> List[str]:
    from PIL import Image
    x = batch.detach().cpu().clamp(0, 1)
    if x.dim() != 4:
        raise ValueError(f"Erzeugte Ausgabe hat die Form {list(x.shape)}, erwartet wird [B, C, H, W]")
    out = []
    for img in x:
        arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
        out.append(_png(Image.fromarray(arr[..., 0] if arr.shape[2] == 1 else arr).resize((160, 160), Image.NEAREST)))
    return out


def _boxes_image(path: str, boxes: List[Dict[str, Any]]) -> str:
    from PIL import Image, ImageDraw
    with Image.open(path) as im:
        image = im.convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for b in boxes:
        cx, cy, bw, bh = b["box"]
        draw.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                       outline=(251, 146, 60), width=max(2, w // 200))
    return _png(image)


def _mask_image(model: Any, path: str) -> Dict[str, Any]:
    from PIL import Image
    size = model.preprocess["pipeline"]["size"]
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((size, size))
    x = (torch.from_numpy(np.asarray(rgb, dtype="float32") / 255.0).permute(2, 0, 1) - 0.5) / 0.5
    model.model.eval()
    with torch.no_grad():
        mask = model.model(x.unsqueeze(0).to(model.device)).argmax(1)[0].cpu().numpy()
    tint = np.asarray(rgb, dtype="float32")
    tint[mask > 0] = tint[mask > 0] * 0.4 + np.array([251, 146, 60]) * 0.6
    share = float((mask > 0).mean())
    return {"predicted": f"{share * 100:.1f} % markiert", "images": [_png(Image.fromarray(tint.astype("uint8")))],
            "inference_ms": 0.0, "task_type": "segmentation"}


def infer(model_dir: str, kind: str, value: Any) -> Dict[str, Any]:
    model = _load(model_dir)
    if kind == "generate":
        return {"predicted": "", "images": _tensor_images(model.generate_images(4)), "inference_ms": 0.0,
                "task_type": model.task_type}
    if kind == "text":
        return model.predict_text(str(value))
    if kind == "tensor":
        wanted = model.preprocess.get("input_shape")
        try:
            return model.predict(torch.tensor(value, dtype=torch.float32))
        except RuntimeError as e:
            raise ValueError(f"Diese Zahlen passen nicht zum Modell. Erwartet wird die Form {wanted}. "
                             "Bilder, Audio und Video als Datei waehlen.") from e
    if kind == "file":
        spec = model.preprocess.get("pipeline") or {}
        if spec.get("kind") == "mask":
            return _mask_image(model, str(value))
        result = model.predict_file(str(value))
        if spec.get("kind") == "boxes":
            result["images"] = [_boxes_image(str(value), result["boxes"])]
        return result
    raise ValueError(f"Unbekannte Eingabe: {kind}")
