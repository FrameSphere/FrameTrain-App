"""Tests fuer den CLIP-Score (plugins/_shared_clip.py).

Aufruf:  python3.11 -m unittest plugins.test_shared_clip   (aus test_engine/)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared_clip import ClipScorer, clip_score_from_embeddings, mean  # noqa: E402

TINY_CLIP = "hf-internal-testing/tiny-random-CLIPModel"


class Formel(unittest.TestCase):
    def test_bekannte_vektoren(self):
        self.assertAlmostEqual(clip_score_from_embeddings([1, 0], [1, 0]), 100.0)
        # cos 60 Grad = 0.5
        self.assertAlmostEqual(clip_score_from_embeddings([1, 0], [0.5, 3 ** 0.5 / 2]), 50.0)
        # Laenge egal, nur Richtung
        self.assertAlmostEqual(clip_score_from_embeddings([3, 4], [6, 8]), 100.0)
        # negative Aehnlichkeit wird auf 0 begrenzt (wie torchmetrics)
        self.assertEqual(clip_score_from_embeddings([1, 0], [-1, 0]), 0.0)
        self.assertEqual(clip_score_from_embeddings([0, 0], [1, 0]), 0.0)

    def test_wie_torchmetrics_formel(self):
        import torch
        a, b = torch.randn(64, generator=torch.Generator().manual_seed(1)), torch.randn(64)
        want = max(100 * torch.nn.functional.cosine_similarity(a, b, dim=0).item(), 0)
        self.assertAlmostEqual(clip_score_from_embeddings(a.tolist(), b.tolist()), want, places=4)

    def test_mittel_ohne_none(self):
        self.assertEqual(mean([10.0, None, 20.0]), 15.0)
        self.assertIsNone(mean([None]))


class Laden(unittest.TestCase):
    def test_fehlendes_modell_liefert_grund_statt_fehler(self):
        scorer, reason = ClipScorer.load("gibt-es-nicht/clip-xyz")
        self.assertIsNone(scorer)
        self.assertIn("lokal nicht vorhanden", reason)
        self.assertIn("gibt-es-nicht/clip-xyz", reason)

    def test_winziges_clip_rechnet(self):
        scorer, reason = ClipScorer.load(TINY_CLIP)
        if scorer is None:
            self.skipTest(reason)
        from PIL import Image
        img = Image.new("RGB", (32, 32), (200, 30, 30))
        s1 = scorer.score(img, "a red square")
        self.assertGreaterEqual(s1, 0.0)
        self.assertLessEqual(s1, 100.0)
        self.assertAlmostEqual(s1, scorer.score(img, "a red square"), places=5)  # deterministisch


if __name__ == "__main__":
    unittest.main()
