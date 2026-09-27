"""Tests fuer den YOLO-Lab-Server (ohne Ultralytics-Import).

Aufruf:  python3 -m train_engine.plugins.yolo.test_yolo_inference_server
         (aus src-tauri/python/)

Geprueft wird das, was ohne geladenes Modell pruefbar ist: das Finden der
Gewichte und die Kurzfassung, die im Labor in der Ergebnisspalte und im
CSV-Export landet.
"""
import tempfile
import unittest
from pathlib import Path

from train_engine.plugins.yolo.yolo_inference_server import YoloInferenceServer, extract_boxes, find_weights


class FindWeightsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bevorzugt_best_vor_anderen(self):
        # Ein Trainingslauf laesst best.pt und last.pt liegen; gemeint ist best.
        (self.root / "last.pt").write_bytes(b"x")
        (self.root / "best.pt").write_bytes(b"x")
        self.assertEqual(find_weights(self.root).name, "best.pt")

    def test_findet_gewichte_im_weights_unterordner(self):
        # Ultralytics legt sie so ab: runs/train/exp/weights/best.pt
        (self.root / "weights").mkdir()
        (self.root / "weights" / "best.pt").write_bytes(b"x")
        self.assertEqual(find_weights(self.root).name, "best.pt")

    def test_nimmt_beliebige_pt_wenn_kein_standardname_passt(self):
        (self.root / "yolov8n.pt").write_bytes(b"x")
        self.assertEqual(find_weights(self.root).name, "yolov8n.pt")

    def test_datei_statt_ordner(self):
        f = self.root / "mein_modell.pt"
        f.write_bytes(b"x")
        self.assertEqual(find_weights(f), f)

    def test_ohne_gewichte_klare_meldung(self):
        with self.assertRaises(FileNotFoundError):
            find_weights(self.root)


class SummaryTest(unittest.TestCase):
    def test_ohne_treffer(self):
        self.assertEqual(YoloInferenceServer._summary([], {}), "Keine Objekte")

    def test_zaehlt_boxen_nennt_klassen(self):
        boxes = [{"label": "Sky"}, {"label": "Sky"}, {"label": "Tree"}]
        best = {"Sky": 0.86, "Tree": 0.80}
        self.assertEqual(YoloInferenceServer._summary(boxes, best), "3x Sky, Tree")

    def test_kuerzt_lange_klassenlisten(self):
        boxes = [{"label": f"K{i}"} for i in range(6)]
        best = {f"K{i}": 0.5 for i in range(6)}
        # Drei Namen plus Zaehler – die Spalte im Labor ist schmal.
        self.assertEqual(YoloInferenceServer._summary(boxes, best), "6x K0, K1, K2 +3")


class _L(list):
    """Liste mit tolist(), wie ein Tensor."""
    def tolist(self):
        return [list(x) if isinstance(x, (list, tuple)) else x for x in self]


class _Box:
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = cls, conf, [xyxy]


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class ExtractBoxesTest(unittest.TestCase):
    """Masken, Keypoints und gedrehte Boxen landen im Labor-Ergebnis."""

    def test_detect_wie_bisher(self):
        r = _Obj(boxes=[_Box(0, 0.9, (1, 2, 3, 4))], obb=None, masks=None, keypoints=None)
        (b,) = extract_boxes(r, {0: "person"})
        self.assertEqual((b["label"], b["x1"], b["y2"]), ("person", 1.0, 4.0))
        self.assertNotIn("polygon", b)
        self.assertNotIn("keypoints", b)

    def test_segment_und_pose(self):
        contour = _L([(float(i), float(i)) for i in range(200)])
        r = _Obj(boxes=[_Box(0, 0.8, (0, 0, 9, 9))], obb=None,
                 masks=_Obj(xy=[contour]),
                 keypoints=_Obj(xy=_L([[(1, 2), (3, 4)]]), conf=_L([[0.9, 0.1]])))
        (b,) = extract_boxes(r, {0: "person"})
        self.assertLessEqual(len(b["polygon"]), 64, "Umriss fuer die Anzeige ausgeduennt")
        self.assertEqual(b["keypoints"], [[1.0, 2.0, 0.9], [3.0, 4.0, 0.1]])

    def test_obb_liefert_vier_ecken(self):
        obb = _L([_Box(1, 0.7, (0, 0, 4, 4))])
        obb.xyxyxyxy = _L([[(0, 1), (1, 0), (4, 3), (3, 4)]])
        r = _Obj(boxes=None, obb=obb, masks=None, keypoints=None)
        (b,) = extract_boxes(r, {1: "schiff"})
        self.assertEqual(b["label"], "schiff")
        self.assertEqual(len(b["polygon"]), 4)


if __name__ == "__main__":
    unittest.main()
