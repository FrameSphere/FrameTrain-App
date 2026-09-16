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

from train_engine.plugins.yolo.yolo_inference_server import YoloInferenceServer, find_weights


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


if __name__ == "__main__":
    unittest.main()
