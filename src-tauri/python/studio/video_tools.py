#!/usr/bin/env python3
"""
FrameTrain - Video-Hilfen fuer die Datensatz-Werkstatt
======================================================
Zwei Befehle, Ausgabe zeilenweise als JSON auf stdout:

    probe --video <pfad>
        {"type": "done", "duration": 12.4, "fps": 29.97, "frames": 372,
         "width": 1920, "height": 1080}

    cut --jobs <jobs.json>
        jobs.json: [{"src": "...", "start": 0.0, "end": 4.0, "out": ".../a.mp4"}, ...]
        {"type": "progress", "current": 3, "total": 20}
        {"type": "done", "written": 20, "failed": []}

Warum schneiden erst beim Export: beim Labeln verweist ein Abschnitt nur auf
Start und Ende im Originalvideo. Wird ein Abschnitt verschoben oder geteilt,
entsteht keine neue Datei; erst der fertige Datensatz braucht einzelne Clips,
weil das Training je Datei eine Klasse liest.

OpenCV schreibt mp4v (MPEG-4 Part 2). Das spielt nicht jeder Browser ab, das
Training liest es aber ueber dieselbe Bibliothek ohne Umweg ueber ffmpeg.
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


def need_cv2():
    try:
        import cv2  # noqa: F401
        return cv2
    except ImportError:
        emit({"type": "error", "message":
              "OpenCV fehlt. Es wird mit dem YOLO- oder Video-Plugin installiert "
              "(Einstellungen > Plugins) oder mit: pip install opencv-python"})
        sys.exit(1)


def probe(video: Path) -> dict:
    cv2 = need_cv2()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Video laesst sich nicht oeffnen: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frames / fps if fps > 0 else 0.0
    if duration <= 0:
        # Manche Container melden keine Bildzahl — dann bis zum Ende lesen.
        n = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            n += 1
        frames = n
        duration = n / fps if fps > 0 else 0.0
    cap.release()
    return {"duration": round(duration, 3), "fps": fps, "frames": frames,
            "width": width, "height": height}


def cut_one(cv2, src: str, start: float, end: float, out: str) -> None:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError(f"Video laesst sich nicht oeffnen: {src}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    first = int(round(max(0.0, start) * fps))
    last = int(round(end * fps)) if end and end > start else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Clip laesst sich nicht schreiben: {out}")
    idx = first
    written = 0
    while last is None or idx < last:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        written += 1
        idx += 1
    writer.release()
    cap.release()
    if written == 0:
        Path(out).unlink(missing_ok=True)
        raise ValueError(f"Keine Bilder zwischen {start:.2f} s und {end:.2f} s in {src}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--video", required=True)
    c = sub.add_parser("cut")
    c.add_argument("--jobs", required=True)
    args = ap.parse_args()

    try:
        if args.cmd == "probe":
            emit({"type": "done", **probe(Path(args.video))})
            return 0

        cv2 = need_cv2()
        jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
        failed = []
        for i, job in enumerate(jobs):
            if i % 5 == 0:
                emit({"type": "progress", "current": i, "total": len(jobs)})
            try:
                cut_one(cv2, job["src"], float(job.get("start") or 0.0),
                        float(job.get("end") or 0.0), job["out"])
            except Exception as e:  # ein kaputter Abschnitt bricht nicht alles ab
                failed.append({"out": job.get("out"), "message": str(e)})
        emit({"type": "done", "written": len(jobs) - len(failed), "failed": failed})
        return 0
    except Exception as e:
        emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
