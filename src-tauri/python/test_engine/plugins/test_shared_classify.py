"""Tests fuer die Klassifikations-Kennzahlen (plugins/_shared_classify.py).

Aufruf:  python3.11 -m unittest plugins.test_shared_classify   (aus test_engine/)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared_classify import ERROR_LABEL, classification_metrics  # noqa: E402


class BekannteWerte(unittest.TestCase):
    # 6 Beispiele, 3 Klassen, von Hand gerechnet:
    #   katze: tp=2 fp=1 fn=0 -> P=2/3 R=1   F1=0.8
    #   hund:  tp=1 fp=0 fn=1 -> P=1   R=1/2 F1=2/3
    #   vogel: tp=1 fp=1 fn=1 -> P=1/2 R=1/2 F1=1/2
    Y = ["katze", "katze", "hund", "hund", "vogel", "vogel"]
    P = ["katze", "katze", "hund", "vogel", "vogel", "katze"]

    def setUp(self):
        self.m = classification_metrics(self.Y, self.P)

    def test_accuracy_und_macro(self):
        self.assertAlmostEqual(self.m["accuracy"], 4 / 6, places=4)
        self.assertAlmostEqual(self.m["macro_f1"], (0.8 + 2 / 3 + 0.5) / 3, places=4)
        self.assertAlmostEqual(self.m["macro_precision"], (2 / 3 + 1 + 0.5) / 3, places=4)
        self.assertAlmostEqual(self.m["macro_recall"], (1 + 0.5 + 0.5) / 3, places=4)

    def test_je_klasse(self):
        pc = self.m["per_class"]
        self.assertEqual(pc["katze"], {"precision": 0.6667, "recall": 1.0, "f1": 0.8, "support": 2})
        self.assertEqual(pc["hund"]["recall"], 0.5)
        self.assertEqual(pc["vogel"]["f1"], 0.5)
        # Altes Feld des Textplugins bleibt erhalten.
        self.assertEqual(self.m["per_class_f1"]["katze"], 0.8)

    def test_konfusionsmatrix(self):
        cm = self.m["confusion_matrix"]
        self.assertEqual(cm["labels"], ["hund", "katze", "vogel"])
        # Zeile = erwartet, Spalte = vorhergesagt
        self.assertEqual(cm["matrix"], [[1, 0, 1], [0, 2, 0], [0, 1, 1]])
        self.assertFalse(cm["truncated"])
        self.assertEqual(self.m["top_confusions"][0], {"expected": "hund", "predicted": "vogel", "count": 1})
        self.assertEqual(len(self.m["top_confusions"]), 2)

    def test_gleich_wie_sklearn(self):
        try:
            from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        except ImportError:
            self.skipTest("sklearn fehlt")
        p, r, f, _ = precision_recall_fscore_support(self.Y, self.P, average="macro", zero_division=0)
        self.assertAlmostEqual(self.m["macro_f1"], f, places=4)
        self.assertAlmostEqual(self.m["macro_precision"], p, places=4)
        self.assertAlmostEqual(self.m["macro_recall"], r, places=4)
        labels = ["hund", "katze", "vogel"]
        self.assertEqual(self.m["confusion_matrix"]["matrix"],
                         confusion_matrix(self.Y, self.P, labels=labels).tolist())


class Randfaelle(unittest.TestCase):
    def test_ohne_labels_leer(self):
        self.assertEqual(classification_metrics([None, None], ["a", "b"]), {})

    def test_error_ist_keine_klasse_aber_ein_fehler(self):
        m = classification_metrics(["a", "a", "b"], ["a", ERROR_LABEL, "b"])
        self.assertNotIn(ERROR_LABEL, m["per_class"])
        self.assertEqual(m["per_class"]["a"]["recall"], 0.5)
        self.assertEqual(m["per_class"]["a"]["support"], 2)
        self.assertAlmostEqual(m["accuracy"], 2 / 3, places=4)
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            return
        self.assertAlmostEqual(
            m["macro_f1"],
            f1_score(["a", "a", "b"], ["a", ERROR_LABEL, "b"], labels=["a", "b"], average="macro", zero_division=0),
            places=4)

    def test_vorhergesagte_fremdklasse_zaehlt_mit(self):
        # Wie sklearn: Klasse, die nur vorhergesagt wird, geht mit F1 0 in den Schnitt ein.
        m = classification_metrics(["a", "a"], ["a", "x"])
        self.assertIn("x", m["per_class"])
        self.assertEqual(m["per_class"]["x"]["support"], 0)
        self.assertAlmostEqual(m["macro_f1"], (2 / 3 + 0) / 2, places=4)

    def test_grosse_matrix_wird_auf_verwechselte_klassen_begrenzt(self):
        # 30 Klassen, alle richtig ausser k03->k07 (3x) und k10->k11 (1x).
        y, p = [], []
        for i in range(30):
            y += [f"k{i:02d}"] * 2
            p += [f"k{i:02d}"] * 2
        y += ["k03"] * 3 + ["k10"]
        p += ["k07"] * 3 + ["k11"]
        m = classification_metrics(y, p, max_matrix_classes=4)
        cm = m["confusion_matrix"]
        self.assertTrue(cm["truncated"])
        self.assertEqual(cm["total_classes"], 30)
        self.assertEqual(cm["labels"], ["k03", "k07", "k10", "k11"])
        self.assertEqual(cm["matrix"][0][1], 3)
        self.assertEqual(m["top_confusions"][0], {"expected": "k03", "predicted": "k07", "count": 3})


if __name__ == "__main__":
    unittest.main()
