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

    def test_kurze_laeufe_bekommen_messpunkte(self):
        def state(gesamt, schritte):
            s = type("S", (), {})()
            s.max_steps, s.logging_steps = gesamt, schritte
            hft.logge_kurze_laeufe(s)
            return s.logging_steps
        self.assertEqual(state(4, 10), 1)      # 7 Aufnahmen, Batch 2
        self.assertEqual(state(42, 10), 4)
        self.assertEqual(state(1000, 10), 10)  # lange Laeufe bleiben
        self.assertEqual(state(0, 10), 10)     # unbekannte Laenge


    def test_textklassifikation_meldet_letzten_loss_statt_mittelwert(self):
        # Live 1.11.2: Analyse zeigte 0.8513 (Mittelwert des Laufs) statt der
        # zuletzt gemessenen 0.5566.
        from types import SimpleNamespace
        from plugins.seq_classification import plugin as seq

        def validiere(gesehen):
            p = seq.Plugin.__new__(seq.Plugin)
            p.config = SimpleNamespace(epochs=1)
            p._trainer = SimpleNamespace(
                evaluate=lambda: {"eval_loss": 0.6},
                state=SimpleNamespace(best_epoch=1, log_history=[], epoch=1.0))
            p._train_result = SimpleNamespace(global_step=6, training_loss=0.8513)
            p._start_time = 0
            p._last_train_loss, p._train_loss_seen = 0.5566, gesehen
            p.num_labels, p.problem_type = 2, seq.SINGLE_LABEL
            p.train_dataset = p.eval_dataset = None
            return p.validate()["final_train_loss"]

        self.assertAlmostEqual(validiere(True), 0.5566)
        self.assertAlmostEqual(validiere(False), 0.8513)  # ohne Messpunkt: Notbehelf


if __name__ == "__main__":
    unittest.main(verbosity=2)
