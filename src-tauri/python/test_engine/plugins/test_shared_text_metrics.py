"""Tests fuer Kennzahlen freier Text-Antworten (plugins/_shared_text_metrics.py).

Aufruf:  python3.11 -m unittest plugins.test_shared_text_metrics   (aus test_engine/)
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # ft_data
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from _shared_text_metrics import (  # noqa: E402
    character_error_rate, free_text_scores, levenshtein, looks_like_ocr, normalise, tolerant_match,
)


class Tolerant(unittest.TestCase):
    def test_normalisierung(self):
        self.assertEqual(normalise("  The Red, Cat! "), "red cat")
        self.assertEqual(normalise("Eine rote Katze."), "rote katze")
        self.assertEqual(normalise("the"), "the")  # einzelnes Wort bleibt

    def test_treffer(self):
        self.assertTrue(tolerant_match("Red cat.", "red cat"))
        self.assertTrue(tolerant_match("It is a red cat on a mat", "red cat"))
        self.assertTrue(tolerant_match("Two", "2") is False)
        # Teilwort zaehlt nicht: "cat" ist nicht in "category"
        self.assertFalse(tolerant_match("category", "cat"))
        self.assertFalse(tolerant_match("blue cat", "red cat"))


class Cer(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abc", "abc"), 0)

    def test_cer_ueber_dataset_summiert(self):
        # (3 + 1) Editierschritte / (7 + 4) Referenzzeichen
        self.assertAlmostEqual(character_error_rate(["kitten", "Hallo"], ["sitting", "Hall"]), 4 / 11)
        self.assertEqual(character_error_rate(["x"], [""]), None)

    def test_cer_wie_jiwer(self):
        try:
            import jiwer
        except ImportError:
            self.skipTest("jiwer fehlt")
        p, r = ["STOP 30 km/h", "Ausfahrt"], ["STOP 50 km/h", "Ausfahrt 12"]
        self.assertAlmostEqual(character_error_rate(p, r), jiwer.cer(r, p), places=6)


class Gesamt(unittest.TestCase):
    P = ["A red cat.", "It is a dog", "blue"]
    R = ["a red cat", "dog", "green"]

    def test_kennzahlen(self):
        s = free_text_scores(self.P, self.R, prompts=["What animal?"] * 3)
        self.assertAlmostEqual(s["exact_match"], 1 / 3)
        self.assertAlmostEqual(s["tolerant_match"], 2 / 3)
        # ROUGE-L: 1.0, F1("it is a dog","dog") = 2*(1/4*1)/(1/4+1) = 0.4, 0.0
        self.assertAlmostEqual(s["rougeL"], (1.0 + 0.4 + 0.0) / 3)
        self.assertNotIn("cer", s)  # keine Texterkennung

    def test_cer_nur_bei_texterkennung(self):
        self.assertTrue(looks_like_ocr(["Read the text in the image."] * 2 + ["What?"]))
        self.assertFalse(looks_like_ocr(["What color is the car?"]))
        s = free_text_scores(["ABC"], ["ABD"], prompts=["Transcribe the sign"])
        self.assertAlmostEqual(s["cer"], 1 / 3)
        self.assertIn("cer", free_text_scores(["a"], ["b"], ocr=True))
        self.assertNotIn("cer", free_text_scores(["a"], ["b"], prompts=["ocr"], ocr=False))


if __name__ == "__main__":
    unittest.main()
