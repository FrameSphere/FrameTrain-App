"""Vorschaubild fuer KI-Assistenten: Bild (oder Video-Frame) mit eingezeichneten Boxen.

Aufruf:  python -m ft_data.preview '{"path": "...", "boxes": [{"cls": 0, "x": .5, "y": .5, "w": .2, "h": .3}],
          "classes": ["person", "car"], "max_size": 1024}'

Ohne "boxes" liest das Skript eine YOLO-Labeldatei neben dem Bild (images/ -> labels/)
und die Klassenliste aus data.yaml/classes.txt in den Ordnern darueber.
Ausgabe: {"type": "result", "mime": "image/jpeg", "data": "<base64>", "width", "height", "boxes": n}.

Warum: Claude und ChatGPT verstehen Bilder sehr gut, bekamen aber nie eines zu sehen.
Mit der Vorschau koennen sie Labels pruefen, fehlende Boxen ergaenzen und schlechte
Bilder aussortieren.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .detection import IMAGE_EXTS, label_path_for
from .fetch import VIDEO_EXTS, FetchError, emit, _parse_names

PALETTE = [(255, 99, 71), (46, 204, 113), (52, 152, 219), (241, 196, 15), (155, 89, 182),
           (26, 188, 156), (230, 126, 34), (236, 72, 153), (149, 165, 166), (0, 206, 209)]


def _classes_near(path: Path) -> List[str]:
    for d in list(path.parents)[:4]:
        for name in ("data.yaml", "dataset.yaml", "classes.txt", "obj.names"):
            f = d / name
            if f.exists():
                text = f.read_text(encoding="utf-8", errors="ignore")
                names = _parse_names(text) if f.suffix in (".yaml", ".yml") else [l.strip() for l in text.splitlines() if l.strip()]
                if names:
                    return names
    return []


def _boxes_from_label(path: Path) -> List[Dict[str, float]]:
    lp = label_path_for(path)
    out = []
    if lp.exists():
        for line in lp.read_text(errors="ignore").splitlines():
            p = line.split()
            if len(p) >= 5 and p[0].isdigit():
                out.append({"cls": int(p[0]), "x": float(p[1]), "y": float(p[2]), "w": float(p[3]), "h": float(p[4])})
    return out


def _video_frame(path: Path, at: float):
    try:
        import cv2  # noqa: WPS433
    except Exception:
        raise FetchError("Video preview needs OpenCV (cv2) in the FrameTrain Python environment.")
    cap = cv2.VideoCapture(str(path))
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * min(max(at, 0.0), 0.99)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FetchError("Could not read a frame from the video.")
    from PIL import Image
    return Image.fromarray(frame[:, :, ::-1])


def render(req: Dict[str, Any]) -> Dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont
    path = Path(req["path"])
    if not path.is_file():
        raise FetchError("File not found.")
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        im = Image.open(path).convert("RGB")
    elif ext in VIDEO_EXTS:
        im = _video_frame(path, float(req.get("at", 0.5)))
    else:
        raise FetchError("Preview is available for images and videos only.")
    boxes: Optional[List[Dict[str, Any]]] = req.get("boxes")
    if boxes is None and ext in IMAGE_EXTS:
        boxes = _boxes_from_label(path)
    classes = req.get("classes") or (_classes_near(path) if boxes else [])
    max_size = int(req.get("max_size") or 1024)
    im.thumbnail((max_size, max_size))
    w, h = im.size
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(12, w // 50))
    except TypeError:
        font = ImageFont.load_default()
    for i, b in enumerate(boxes or []):
        cid = int(b.get("cls", 0))
        color = PALETTE[cid % len(PALETTE)]
        x0, y0 = (b["x"] - b["w"] / 2) * w, (b["y"] - b["h"] / 2) * h
        x1, y1 = (b["x"] + b["w"] / 2) * w, (b["y"] + b["h"] / 2) * h
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, w // 300))
        name = classes[cid] if cid < len(classes) else str(cid)
        label = f"{i + 1}:{name}"
        tw = draw.textlength(label, font=font)
        th = max(12, w // 50) + 4
        ty = y0 - th if y0 - th > 0 else y0
        draw.rectangle([x0, ty, x0 + tw + 6, ty + th], fill=color)
        draw.text((x0 + 3, ty + 1), label, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return {"mime": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            "width": w, "height": h, "boxes": len(boxes or []), "classes": classes}


def main(argv: List[str]) -> int:
    try:
        req = json.loads(argv[1]) if len(argv) > 1 else json.load(sys.stdin)
        emit("result", **render(req))
        return 0
    except FetchError as e:
        emit("error", message=str(e))
        return 2
    except Exception as e:  # noqa: BLE001
        emit("error", message=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
