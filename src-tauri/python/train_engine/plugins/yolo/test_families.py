"""Ultralytics-Familien im YOLO-Plugin: RT-DETR, YOLO-World, YOLOE, v10/12/26.

Hintergrund: rtdetr-l.pt galt nicht als YOLO-Gewicht, YOLO-World brach mitten
im Training mit "No module named 'clip'" ab, und RT-DETR mit den YOLO-
Standardwerten (bzw. AdamW mit warmup_bias_lr 0.1) endete bei mAP50 0.

Aufruf:  python3 plugins/yolo/test_families.py   (aus train_engine/)
"""
import sys, tempfile, types, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.config import TrainingConfig                      # noqa: E402
from plugins.yolo.plugin import YOLOPlugin, task_from_weights_name  # noqa: E402
from ft_data import ultralytics_family as uf                 # noqa: E402


class FamilyFromNameTest(unittest.TestCase):
    def test_familien(self):
        cases = {
            "rtdetr-l.pt": "rtdetr", "rtdetr-x.pt": "rtdetr",
            "yolov8s-world.pt": "yolo_world", "yolov8l-worldv2.pt": "yolo_world",
            "yoloe-11s-seg.pt": "yoloe", "yolov10n.pt": "yolov10", "yolo12s.pt": "yolo12",
            "yolo26n.pt": "yolo26", "yolo11n.pt": "yolo", "yolov8n-seg.pt": "yolo",
            "model.pt": None, "best.pt": None,
        }
        for name, fam in cases.items():
            self.assertEqual(uf.family_from_name(name), fam, name)

    def test_rtdetr_ist_detect(self):
        self.assertEqual(task_from_weights_name("rtdetr-l.pt"), "detect")
        self.assertEqual(task_from_weights_name("/x/rtdetr-x.pt"), "detect")

    def test_model_json_gewinnt_vor_dem_namen(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "model.pt").write_bytes(b"x")
            (Path(d) / "model.json").write_text('{"family": "rtdetr", "task": "detect"}')
            self.assertEqual(uf.resolve_family(str(Path(d) / "model.pt")), "rtdetr")
            # Aeltere model.json ohne family: Name des Basismodells zaehlt.
            (Path(d) / "model.json").write_text('{"base_model": "/w/yolov10n.pt"}')
            self.assertEqual(uf.resolve_family(str(Path(d) / "model.pt")), "yolov10")

    def test_family_of_model_an_der_klasse(self):
        RTDETR = type("RTDETR", (), {})
        self.assertEqual(uf.family_of_model(RTDETR()), "rtdetr")
        self.assertEqual(uf.family_of_model(type("YOLOWorld", (), {})()), "yolo_world")


class LoaderTest(unittest.TestCase):
    """load_ultralytics nimmt die passende Klasse (ohne echte Gewichte)."""

    def setUp(self):
        fake = types.ModuleType("ultralytics")
        for cls in ("YOLO", "RTDETR", "YOLOWorld", "YOLOE"):
            setattr(fake, cls, type(cls, (), {"__init__": lambda self, w, _c=cls: setattr(self, "w", w)}))
        patcher = mock.patch.dict(sys.modules, {"ultralytics": fake})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_klassen(self):
        self.assertEqual(type(uf.load_ultralytics("rtdetr-l.pt")).__name__, "RTDETR")
        self.assertEqual(type(uf.load_ultralytics("yolov8s-world.pt")).__name__, "YOLOWorld")
        self.assertEqual(type(uf.load_ultralytics("yoloe-11s-seg.pt")).__name__, "YOLOE")
        self.assertEqual(type(uf.load_ultralytics("yolo12n.pt")).__name__, "YOLO")
        # model.pt eines trainierten RT-DETR: Familie aus dem Aufrufer
        self.assertEqual(type(uf.load_ultralytics("model.pt", "rtdetr")).__name__, "RTDETR")


class BlockerTest(unittest.TestCase):
    def test_world_ohne_clip(self):
        with mock.patch.object(uf, "_has_module", return_value=False):
            title, detail = uf.training_blocker("yolov8s-world.pt")
        self.assertIn("clip", title)
        self.assertIn("pip install", detail)

    def test_world_mit_clip_laeuft(self):
        with mock.patch.object(uf, "_has_module", return_value=True):
            self.assertIsNone(uf.training_blocker("yolov8s-world.pt"))

    def test_yoloe_und_semantic(self):
        self.assertIn("YOLOE", uf.training_blocker("yoloe-11s-seg.pt")[0])
        self.assertIn("Semantische", uf.training_blocker("yolo26n-sem.pt")[0])

    def test_trainierbare_familien(self):
        for w in ("rtdetr-l.pt", "yolov10n.pt", "yolo12n.pt", "yolo26n.pt", "yolo11n.pt"):
            self.assertIsNone(uf.training_blocker(w), w)


class HyperparamTest(unittest.TestCase):
    def test_warmup_bias_fuer_adam(self):
        self.assertEqual(uf.warmup_bias_lr_for("AdamW"), 0.0)
        self.assertEqual(uf.warmup_bias_lr_for("adam"), 0.0)
        self.assertIsNone(uf.warmup_bias_lr_for("SGD"))
        self.assertIsNone(uf.warmup_bias_lr_for("auto"))
        self.assertEqual(uf.warmup_bias_lr_for("AdamW", 0.05), 0.05)

    def _plugin(self, d, weights, pc=None):
        (Path(d) / weights).write_bytes(b"x")
        cfg = TrainingConfig()
        cfg.model_path = d
        cfg.plugin_config = {"optimizer": "SGD", "lr0": 0.01, **(pc or {})}
        p = YOLOPlugin(cfg)
        p.yolo_model = str(Path(d) / weights)
        return p

    def test_rtdetr_bekommt_auto_statt_sgd(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._plugin(d, "rtdetr-l.pt")
            self.assertTrue(p._check_family())
            self.assertEqual((p.family, p.optimizer_name), ("rtdetr", "auto"))

    def test_eigene_werte_bleiben(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._plugin(d, "rtdetr-l.pt", {"optimizer": "AdamW", "lr0": 0.0002})
            self.assertTrue(p._check_family())
            self.assertEqual((p.optimizer_name, p.lr0), ("AdamW", 0.0002))

    def test_yolo_bleibt_sgd(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._plugin(d, "yolo11n.pt")
            self.assertTrue(p._check_family())
            self.assertEqual((p.family, p.optimizer_name, p.lr0), ("yolo", "SGD", 0.01))

    def test_blocker_bricht_setup_ab(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._plugin(d, "yoloe-11s-seg.pt")
            with mock.patch("plugins.yolo.plugin.MessageProtocol.error") as err:
                self.assertFalse(p._check_family())
            self.assertIn("YOLOE", err.call_args[0][0])



class GeraetAufAppleSilicon(unittest.TestCase):
    def test_rtdetr_trainiert_auf_cpu_statt_mps(self):
        from ft_data.ultralytics_family import mps_or_cpu
        self.assertEqual(mps_or_cpu("rtdetr"), "cpu")
        self.assertEqual(mps_or_cpu("yolo"), "mps")

if __name__ == "__main__":
    unittest.main(verbosity=2)
