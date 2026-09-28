"""Tests fuer die Schutzfunktionen in core.hf_training (ohne Modell-Download).

Aufruf: python3 -m unittest test_hf_safety   (aus src-tauri/python/train_engine/)
"""
import unittest

import torch

from core import hf_training as hft


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_spec_embed = torch.nn.Parameter(torch.full((4,), float("nan")))
        self.lin = torch.nn.Linear(3, 2)
        self.norm = torch.nn.LayerNorm(2)


class HfSafetyTest(unittest.TestCase):
    def test_uninitialisiertes_masked_spec_embed_wird_repariert(self):
        m = _Tiny()
        self.assertEqual(hft.repair_uninitialized_params(m), ["masked_spec_embed"])
        self.assertTrue(torch.isfinite(m.masked_spec_embed).all())
        self.assertTrue(((m.masked_spec_embed >= 0) & (m.masked_spec_embed <= 1)).all())

    def test_riesige_werte_gelten_als_nicht_initialisiert(self):
        m = _Tiny()
        with torch.no_grad():
            m.masked_spec_embed.fill_(3e37)
        self.assertTrue(hft.repair_uninitialized_params(m))

    def test_gesunder_parameter_bleibt(self):
        m = _Tiny()
        with torch.no_grad():
            m.masked_spec_embed.fill_(0.5)
        self.assertEqual(hft.repair_uninitialized_params(m), [])
        self.assertTrue(torch.equal(m.masked_spec_embed, torch.full((4,), 0.5)))

    def test_einzelner_nan_gradient_wird_bereinigt_nicht_verworfen(self):
        m = _Tiny()
        for p in m.parameters():
            p.grad = torch.ones_like(p)
        m.norm.bias.grad[0] = float("nan")

        class T:
            pass
        self.assertFalse(hft.sanitize_nan_grads(T(), m))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertEqual(float(m.lin.weight.grad.sum()), 6.0)

    def test_viele_nan_gradienten_verwerfen_den_schritt(self):
        m = _Tiny()
        for p in m.parameters():
            p.grad = torch.full_like(p, float("nan"))
        self.assertTrue(hft.sanitize_nan_grads(type("T", (), {})(), m))
        self.assertTrue(all(p.grad is None for p in m.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
