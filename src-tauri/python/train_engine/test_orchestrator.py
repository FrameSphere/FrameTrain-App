"""Prueft, dass ein gescheiterter Plugin-Schritt kein complete-Event erzeugt.

Hintergrund: YOLOPlugin.train() gab bei "images not found" False zurueck, der
Orchestrator lief aber weiter bis complete. Die App zeigte "Training
erfolgreich abgeschlossen" mit Epoch 0/0.

Aufruf:  python3 test_orchestrator.py   (aus train_engine/)
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_engine                          # noqa: E402
from core.config import TrainingConfig       # noqa: E402


class FakePlugin:
    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.is_stopped = False

    def _step(self, name):
        return False if self.fail_at == name else True

    def setup(self):       return self._step("setup")
    def load_data(self):   return self._step("load_data")
    def build_model(self): return self._step("build_model")
    def train(self):       return self._step("train")
    def validate(self):    return {}
    def export(self):      return False if self.fail_at == "export" else "/tmp/out"
    def stop(self):        self.is_stopped = True


def run_with(plugin):
    cfg = TrainingConfig()
    cfg.output_path = tempfile.mkdtemp()
    orig = train_engine.load_plugin
    train_engine.load_plugin = lambda _cfg: plugin
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            train_engine.Orchestrator(cfg).run()
    except SystemExit:
        pass
    finally:
        train_engine.load_plugin = orig
    types = []
    for line in buf.getvalue().splitlines():
        try:
            types.append(json.loads(line).get("type"))
        except ValueError:
            pass
    return types


class OrchestratorTest(unittest.TestCase):
    def test_erfolg_meldet_complete(self):
        self.assertIn("complete", run_with(FakePlugin()))

    def test_gescheiterte_schritte_melden_kein_complete(self):
        for step in ("setup", "load_data", "build_model", "train", "export"):
            with self.subTest(step=step):
                self.assertNotIn("complete", run_with(FakePlugin(fail_at=step)))


if __name__ == "__main__":
    unittest.main()
