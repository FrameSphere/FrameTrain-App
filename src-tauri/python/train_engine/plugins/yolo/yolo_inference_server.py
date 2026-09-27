#!/usr/bin/env python3
"""FrameTrain – Persistenter YOLO-Inferenz-Server fuer das Labor.

Dritter Server neben test_engine/model_server.py (HuggingFace) und
canvas_inference_server.py (Synapse-Netze). Gleiches Protokoll: eine
JSON-Zeile auf stdin rein, eine JSON-Zeile auf stdout raus.

Antwort-Format wie beim HuggingFace-Server (predicted / confidence /
top_predictions / inference_time), damit das Labor Sessions, Notizen,
CSV-Export und Analyse ohne Sonderweg weiterverwendet. Zusaetzlich:

    boxes         Liste der Detektionen mit Pixelkoordinaten
    image_width   Breite des Bildes in Pixeln
    image_height  Hoehe des Bildes in Pixeln
    task          detect / segment / pose / obb / classify

Je Box optional 'polygon' (Maskenumriss bei segment, vier Ecken bei obb) und
'keypoints' ([x, y, sichtbar] bei pose). Ein Klassifikationsmodell liefert
keine Boxen; dort stehen die fuenf wahrscheinlichsten Klassen in
top_predictions.

Ohne die Boxen sagt "Tree 0.80" nicht, *wo* das Modell den Baum sieht —
und genau dafuer geht man im Labor eine Bilderserie durch.
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

# Ultralytics schreibt Fortschritt und Hinweise nach stdout. Auf stdout liegt
# hier aber das Protokoll — deshalb wird das echte stdout festgehalten und
# alles andere nach stderr umgeleitet.
_real_stdout = sys.stdout


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), file=_real_stdout, flush=True)


def emit_error(message: str) -> None:
    emit({"type": "error", "message": message})


def find_weights(model_dir: Path) -> Path:
    """Sucht die Gewichtsdatei im Modell- oder Versionsordner.

    Ein trainiertes YOLO heisst best.pt oder last.pt, ein Basismodell
    yolov8n.pt — verlassen kann man sich auf keinen dieser Namen.
    """
    if model_dir.is_file() and model_dir.suffix in (".pt", ".pth"):
        return model_dir

    preferred = ["best.pt", "last.pt", "weights/best.pt", "weights/last.pt"]
    for rel in preferred:
        candidate = model_dir / rel
        if candidate.exists():
            return candidate

    found = sorted(model_dir.rglob("*.pt"))
    if found:
        return found[0]
    raise FileNotFoundError(
        f"Keine .pt-Gewichte in {model_dir} gefunden — enthaelt der Ordner ein YOLO-Modell?"
    )


_MAX_POLYGON_POINTS = 64


def _thin(points: list, limit: int = _MAX_POLYGON_POINTS) -> list:
    """Maskenumrisse haben oft Hunderte Punkte; fuer die Anzeige reichen 64."""
    if len(points) <= limit:
        return points
    step = len(points) / limit
    return [points[int(i * step)] for i in range(limit)]


def extract_boxes(result, names: dict) -> list:
    """Boxen aller Aufgaben in einem Format.

    detect/segment/pose liefern result.boxes, obb liefert result.obb (gedrehte
    Boxen; x1..y2 ist dann das umschliessende Rechteck, 'polygon' die vier
    Ecken). Segmentierung haengt den Maskenumriss als 'polygon' an, Pose die
    Keypoints als [x, y, sichtbar].
    """
    obb = getattr(result, "obb", None)
    source = obb if obb is not None else getattr(result, "boxes", None)
    if source is None:
        return []
    masks = getattr(result, "masks", None)
    polygons = list(getattr(masks, "xy", None) or []) if masks is not None else []
    kpts = getattr(result, "keypoints", None)
    kpt_xy = kpts.xy.tolist() if kpts is not None and getattr(kpts, "xy", None) is not None else []
    kpt_conf = (kpts.conf.tolist() if kpts is not None and getattr(kpts, "conf", None) is not None
                else [])
    corners = obb.xyxyxyxy.tolist() if obb is not None else []
    out = []
    for i, box in enumerate(source):
        cls_id = int(box.cls)
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        item = {
            "label": names.get(cls_id, str(cls_id)),
            "confidence": round(float(box.conf), 4),
            "x1": round(x1, 1), "y1": round(y1, 1),
            "x2": round(x2, 1), "y2": round(y2, 1),
        }
        if i < len(corners):
            item["polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in corners[i]]
        if i < len(polygons):
            pts = [[round(float(x), 1), round(float(y), 1)] for x, y in polygons[i].tolist()]
            item["polygon"] = _thin(pts)
        if i < len(kpt_xy):
            confs = kpt_conf[i] if i < len(kpt_conf) else [1.0] * len(kpt_xy[i])
            item["keypoints"] = [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                                 for (x, y), c in zip(kpt_xy[i], confs)]
        out.append(item)
    return out


class YoloInferenceServer:
    def __init__(self, model_dir: str, conf: float, iou: float):
        self.model_dir = Path(model_dir)
        self.conf = conf
        self.iou = iou
        self.model = None
        self.names: dict = {}
        self.task = "detect"

    def load(self) -> None:
        weights = find_weights(self.model_dir)
        from ultralytics import YOLO  # erst hier: der Import kostet Sekunden

        with contextlib.redirect_stdout(sys.stderr):
            self.model = YOLO(str(weights))
        self.names = dict(getattr(self.model, "names", {}) or {})
        self.task = getattr(self.model, "task", "detect") or "detect"

    def infer(self, req: dict) -> dict:
        path = (req.get("file_path") or req.get("input") or "").strip()
        if not path:
            raise ValueError("Kein Bildpfad in der Anfrage (file_path).")
        if not Path(path).exists():
            raise FileNotFoundError(f"Bild nicht gefunden: {path}")

        started = time.time()
        with contextlib.redirect_stdout(sys.stderr):
            result = self.model.predict(
                path, conf=self.conf, iou=self.iou, verbose=False
            )[0]
        elapsed = time.time() - started

        names = dict(getattr(result, "names", {}) or self.names)
        height, width = (result.orig_shape if getattr(result, "orig_shape", None)
                         else (0, 0))

        # Klassifikation: keine Boxen, sondern eine Wahrscheinlichkeit je Klasse.
        probs = getattr(result, "probs", None)
        if probs is not None:
            return self._classification(probs, names, width, height, elapsed)

        boxes = extract_boxes(result, names)
        boxes.sort(key=lambda b: b["confidence"], reverse=True)

        # Pro Klasse der beste Treffer – das ist es, was die Balkenanzeige im
        # Labor zeigt. Sieben Boxen mit dreimal "Sky" waeren dort unlesbar.
        best_per_label: dict = {}
        for b in boxes:
            if b["confidence"] > best_per_label.get(b["label"], 0.0):
                best_per_label[b["label"]] = b["confidence"]
        top_predictions = [
            {"label": label, "score": score}
            for label, score in sorted(best_per_label.items(), key=lambda kv: -kv[1])
        ]

        return {
            "type": "result",
            "task": self.task,
            "predicted": self._summary(boxes, best_per_label),
            "confidence": boxes[0]["confidence"] if boxes else None,
            "top_predictions": top_predictions,
            "boxes": boxes,
            "image_width": int(width),
            "image_height": int(height),
            "inference_time": elapsed,
        }

    def _classification(self, probs, names: dict, width: int, height: int, elapsed: float) -> dict:
        data = probs.data.tolist() if hasattr(probs, "data") else list(probs)
        ranked = sorted(range(len(data)), key=lambda i: -float(data[i]))[:5]
        top = [{"label": names.get(i, str(i)), "score": round(float(data[i]), 4)} for i in ranked]
        return {
            "type": "result",
            "task": "classify",
            "predicted": top[0]["label"] if top else "",
            "confidence": top[0]["score"] if top else None,
            "top_predictions": top,
            "boxes": [],
            "image_width": int(width),
            "image_height": int(height),
            "inference_time": elapsed,
        }

    @staticmethod
    def _summary(boxes: list, best_per_label: dict) -> str:
        """Kurzfassung fuer die Ergebnisspalte und den CSV-Export."""
        if not boxes:
            return "Keine Objekte"
        labels = list(best_per_label.keys())
        head = ", ".join(labels[:3])
        if len(labels) > 3:
            head += f" +{len(labels) - 3}"
        return f"{len(boxes)}x {head}"

    def run(self) -> None:
        try:
            self.load()
        except Exception as e:
            emit_error(f"Modell konnte nicht geladen werden: {type(e).__name__}: {e}")
            sys.exit(1)

        emit({
            "type": "ready",
            "modality": "detect",
            "input_kind": "image",
            "task_type": self.task,
            "num_classes": len(self.names),
            "classes": [self.names[k] for k in sorted(self.names)],
        })

        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                req = json.loads(raw_line)
            except json.JSONDecodeError as e:
                emit_error(f"JSON parse: {e}")
                continue

            if req.get("cmd") == "shutdown":
                break

            try:
                emit(self.infer(req))
            except Exception as e:
                emit_error(f"{type(e).__name__}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FrameTrain YOLO Inference Server")
    parser.add_argument("--model-dir", required=True,
                        help="Modell- oder Versionsordner mit den .pt-Gewichten")
    # Bewusst niedrig: das Labor filtert die Boxen ueber einen Regler, und was
    # der Server wegwirft, kann der Regler nicht zurueckholen.
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    if not Path(args.model_dir).exists():
        emit_error(f"Modell-Ordner nicht gefunden: {args.model_dir}")
        sys.exit(1)

    # Sicherheitsnetz gegen Bibliotheken, die direkt auf sys.stdout schreiben.
    sys.stdout = sys.stderr
    YoloInferenceServer(args.model_dir, args.conf, args.iou).run()


if __name__ == "__main__":
    main()
