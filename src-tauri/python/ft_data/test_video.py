"""Tests fuer ft_data.video: dieselbe Bildauswahl in Training, Test und Werkstatt.

Aufruf:  python3 -m ft_data.test_video   (aus src-tauri/python/)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ft_data.media import exts_for
from ft_data.video import frame_indices, read_clip_frames


class VideoTests(unittest.TestCase):
    def test_bildnummern_gleichmaessig_und_im_abschnitt(self):
        self.assertEqual(frame_indices(100, 4), [12, 37, 62, 87])
        idx = frame_indices(100, 4, first=20, last=40)
        self.assertEqual(idx, [22, 27, 32, 37])

    def test_kurzer_clip_wiederholt_bilder(self):
        idx = frame_indices(3, 8)
        self.assertEqual(len(idx), 8)
        self.assertTrue(set(idx) <= {0, 1, 2})

    def test_videoendungen(self):
        self.assertIn(".mp4", exts_for("video"))
        self.assertNotIn(".mp4", exts_for("image"))

    def test_abschnitt_aus_echtem_video(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV fehlt")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "v.mp4"
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
            for f in range(40):
                w.write(np.full((24, 32, 3), f * 5, np.uint8))
            w.release()
            frames = read_clip_frames(path, 4, start=2.0, end=3.0)
            self.assertEqual(len(frames), 4)
            self.assertEqual(frames[0].shape, (24, 32, 3))
            # Bilder 20..29 haben Helligkeit 100..145 — nichts vom Anfang des Videos.
            for f in frames:
                self.assertTrue(95 <= int(f.mean()) <= 150, int(f.mean()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
