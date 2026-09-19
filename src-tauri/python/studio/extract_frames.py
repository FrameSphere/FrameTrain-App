#!/usr/bin/env python3
"""
FrameTrain - Einzelbilder aus einem Video
=========================================
Schreibt jedes n-te Bild eines Videos als JPEG in einen Ordner.

Ausgabe zeilenweise als JSON auf stdout (gleiche Machart wie die
Inferenz-Server):
    {"type": "progress", "current": 120, "total": 3000, "written": 8}
    {"type": "done", "written": 200, "fps": 29.97, "frames": 3000}
    {"type": "error", "message": "..."}

Warum jedes n-te: aufeinanderfolgende Bilder eines Videos sind sich zu
aehnlich, um einzeln etwas beizutragen. Ein Abstand von einer halben Sekunde
liefert Vielfalt statt Wiederholung — und spart das Labeln von 29 fast
identischen Bildern je Sekunde.
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--every-n", type=int, default=15)
    ap.add_argument("--max-frames", type=int, default=2000)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        emit({"type": "error", "message":
              "OpenCV fehlt. Es wird mit dem YOLO-Plugin installiert "
              "(Einstellungen > Plugins) oder mit: pip install opencv-python"})
        return 1

    video = Path(args.video)
    if not video.exists():
        emit({"type": "error", "message": f"Video nicht gefunden: {video}"})
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        emit({"type": "error", "message":
              f"Das Video liess sich nicht oeffnen: {video.name}. "
              "Moeglicherweise fehlt der Codec."})
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    step = max(1, args.every_n)
    stamm = video.stem.replace(" ", "_")

    index = 0
    written = 0
    while written < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            ziel = out_dir / f"{stamm}_{index:06d}.jpg"
            cv2.imwrite(str(ziel), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            written += 1
            if written % 20 == 0:
                emit({"type": "progress", "current": index, "total": total, "written": written})
        index += 1

    cap.release()
    emit({"type": "done", "written": written, "fps": round(fps, 2), "frames": total or index})
    return 0


if __name__ == "__main__":
    sys.exit(main())
