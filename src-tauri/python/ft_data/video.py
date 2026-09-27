"""Videoclips fuer Training, Test und Vorschlaege.

Ein Videoklassifikator (VideoMAE, TimeSformer, ViViT) sieht nicht das ganze
Video, sondern eine feste Zahl Einzelbilder, gleichmaessig ueber den Clip
verteilt. Genau diese Auswahl muss in Training, Test und Werkstatt dieselbe
sein — sonst lernt das Modell auf anderen Bildern, als es spaeter bekommt.
Deshalb liegt sie hier und nirgends sonst.

Gelesen wird mit OpenCV (cv2), das mit dem YOLO- und dem Video-Plugin kommt;
ffmpeg ist nicht noetig.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def frame_indices(total: int, num_frames: int, first: int = 0, last: Optional[int] = None) -> List[int]:
    """Gleichmaessig verteilte Bildnummern in [first, last).

    Hat der Clip weniger Bilder als verlangt, wiederholen sich Nummern — das
    Modell braucht immer genau num_frames Bilder.
    """
    last = total if last is None else min(last, total)
    first = max(0, min(first, max(last - 1, 0)))
    span = max(last - first, 1)
    if num_frames <= 0:
        return []
    return [first + min(int((i + 0.5) * span / num_frames), span - 1) for i in range(num_frames)]


def read_clip_frames(path, num_frames: int, start: Optional[float] = None,
                     end: Optional[float] = None) -> List[np.ndarray]:
    """num_frames RGB-Bilder (H, W, 3, uint8) aus dem Abschnitt [start, end)."""
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - Hinweis fuer den Nutzer
        raise ImportError("OpenCV fehlt. Installiere: pip install opencv-python") from e

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Video laesst sich nicht oeffnen: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        first = int(round((start or 0.0) * fps))
        last = int(round(end * fps)) if end else None
        if total <= 0:
            total = (last or first + num_frames)
        wanted = frame_indices(total, num_frames, first, last)
        frames: List[np.ndarray] = []
        cache = {}
        for idx in wanted:
            if idx in cache:
                frames.append(cache[idx])
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                if frames:
                    frames.append(frames[-1])
                    continue
                raise ValueError(f"Bild {idx} in {Path(path).name} nicht lesbar")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cache[idx] = rgb
            frames.append(rgb)
        return frames
    finally:
        cap.release()
