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


class PluginVertragTest(unittest.TestCase):
    """load_plugin prueft die Pflichtmethoden, bevor das Plugin laeuft."""

    def test_fehlendes_stop_wird_ergaenzt(self):
        class OhneStop:
            def __init__(self, cfg): self.is_stopped = False
            def setup(self): pass
            def load_data(self): pass
            def build_model(self): pass
            def train(self): pass
            def validate(self): return {}
            def export(self): return ""
        cls = train_engine.ensure_plugin_contract(OhneStop, "test")
        p = cls(None)
        p.stop()
        self.assertTrue(p.is_stopped)

    def test_fehlende_pflichtmethode_nennt_sie(self):
        class OhneTrain:
            def setup(self): pass
            def load_data(self): pass
            def build_model(self): pass
            def validate(self): return {}
            def export(self): return ""
        with self.assertRaises(ValueError) as ctx:
            train_engine.ensure_plugin_contract(OhneTrain, "kaputt")
        self.assertIn("train", str(ctx.exception))
        self.assertIn("kaputt", str(ctx.exception))

    def test_abstrakte_basisklasse_wird_erkannt(self):
        from core.plugin_base import TrainPlugin

        class Halb(TrainPlugin):
            def setup(self): pass
        with self.assertRaises(ValueError) as ctx:
            train_engine.ensure_plugin_contract(Halb, "halb")
        self.assertIn("export", str(ctx.exception))

    def test_eigenstaendige_plugins_erben_von_trainplugin(self):
        # Ohne das Verhalten zu aendern: sie bleiben instanziierbar (keine
        # offene abstrakte Methode) und starten ungestoppt.
        import importlib.util
        from core.plugin_base import TrainPlugin
        here = Path(__file__).resolve().parent / "plugins"
        for folder, cls_name in (("yolo", "YOLOPlugin"), ("canvas", "CanvasPlugin"),
                                 ("image_classification", "ImageClassificationPlugin")):
            with self.subTest(plugin=folder):
                spec = importlib.util.spec_from_file_location(
                    f"plugins.{folder}.plugin", here / folder / "plugin.py")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                cls = getattr(mod, cls_name)
                self.assertTrue(issubclass(cls, TrainPlugin))
                self.assertEqual(train_engine.ensure_plugin_contract(cls, folder), cls)
                plugin = cls(TrainingConfig())
                self.assertFalse(plugin.is_stopped)

    def test_load_plugin_liefert_geprueftes_plugin(self):
        cfg = TrainingConfig()
        cfg.task_type = "detect"
        with contextlib.redirect_stdout(io.StringIO()):
            plugin = train_engine.load_plugin(cfg)
        self.assertTrue(callable(plugin.stop))
        cfg.task_type = "gibt_es_nicht"
        with self.assertRaises(ValueError):
            with contextlib.redirect_stdout(io.StringIO()):
                train_engine.load_plugin(cfg)


if __name__ == "__main__":
    unittest.main()
