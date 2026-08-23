"""Prueft die Auswahl der Startgewichte.

Hintergrund: Das Plugin lud fest 'yolov8n.pt' aus dem Netz, waehrend das vom
Nutzer importierte YOLO11 (15 .pt-Dateien) ungenutzt daneben lag.

Aufruf:  python3 plugins/yolo/test_weights.py   (aus train_engine/)
"""
import sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.config import TrainingConfig            # noqa: E402
from plugins.yolo.plugin import YOLOPlugin        # noqa: E402

# Ausschnitt aus Ultralytics/YOLO11, wie es die App herunterlaedt.
YOLO11_FILES = [
    "yolo11l-pose.pt", "yolo11l-seg.pt", "yolo11l.pt",
    "yolo11m-pose.pt", "yolo11m-seg.pt", "yolo11m.pt",
    "yolo11n-pose.pt", "yolo11n-seg.pt", "yolo11n.pt",
    "yolo11s-pose.pt", "yolo11s-seg.pt", "yolo11s.pt",
    "yolo11x-pose.pt", "yolo11x-seg.pt", "yolo11x.pt",
]


def make_plugin(model_dir, plugin_config=None):
    cfg = TrainingConfig()
    cfg.model_path = str(model_dir)
    cfg.plugin_config = plugin_config or {}
    return YOLOPlugin(cfg)


class ResolveWeightsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for name in YOLO11_FILES:
            (self.dir / name).write_bytes(b"x" * (len(name) * 10))
        self.addCleanup(self.tmp.cleanup)

    def test_nimmt_importierte_gewichte_statt_download(self):
        p = make_plugin(self.dir)
        chosen = Path(p._resolve_weights())
        self.assertEqual(chosen.parent, self.dir, "muss aus dem Modellordner kommen")
        self.assertEqual(chosen.name, "yolo11n.pt", "kleinste Detect-Variante erwartet")

    def test_task_suffix_wird_beachtet(self):
        p = make_plugin(self.dir, {"task": "segment"})
        self.assertEqual(Path(p._resolve_weights()).name, "yolo11n-seg.pt")
        p = make_plugin(self.dir, {"task": "pose"})
        self.assertEqual(Path(p._resolve_weights()).name, "yolo11n-pose.pt")

    def test_explizite_auswahl_hat_vorrang(self):
        p = make_plugin(self.dir, {"yolo_model": "yolo11x.pt"})
        self.assertEqual(Path(p._resolve_weights()).name, "yolo11x.pt")

    def test_ohne_gewichte_bleibt_der_download_fallback(self):
        empty = Path(tempfile.mkdtemp())
        p = make_plugin(empty)
        self.assertEqual(p._resolve_weights(), "yolov8n.pt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
