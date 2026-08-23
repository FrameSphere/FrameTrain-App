"""Prueft Startgewichte, Metriken und Split-Groessen des YOLO-Plugins.

Hintergrund: Das Plugin lud fest 'yolov8n.pt' aus dem Netz, waehrend das vom
Nutzer importierte YOLO11 (15 .pt-Dateien) ungenutzt daneben lag.

Aufruf:  python3 plugins/yolo/test_weights.py   (aus train_engine/)
"""
import sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.config import TrainingConfig            # noqa: E402
from plugins.yolo.plugin import YOLOPlugin        # noqa: E402

# Ausschnitt aus Ultralytics/YOLO11, wie es die App herunterlaedt.
YOLO11_FILES = [
    "yolo11l-pose.pt", "yolo11l-seg.pt", "yolo11l.pt",
    "yolo11m-pose.pt", "yolo11m-seg.pt", "yolo11m.pt",
    "yolo11n-pose.pt", "yolo11n-seg.pt", "yolo11n.pt",
    "yolo11s-pose.pt", "yolo11s-seg.pt", "yolo11s.pt",
    "yolo11x-pose.pt", "yolo11x-seg.pt", "yolo11x.pt",
]


def make_plugin(model_dir, plugin_config=None):
    cfg = TrainingConfig()
    cfg.model_path = str(model_dir)
    cfg.plugin_config = plugin_config or {}
    return YOLOPlugin(cfg)


class ResolveWeightsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for name in YOLO11_FILES:
            (self.dir / name).write_bytes(b"x" * (len(name) * 10))
        self.addCleanup(self.tmp.cleanup)

    def test_nimmt_importierte_gewichte_statt_download(self):
        p = make_plugin(self.dir)
        chosen = Path(p._resolve_weights())
        self.assertEqual(chosen.parent, self.dir, "muss aus dem Modellordner kommen")
        self.assertEqual(chosen.name, "yolo11n.pt", "kleinste Detect-Variante erwartet")

    def test_task_suffix_wird_beachtet(self):
        p = make_plugin(self.dir, {"task": "segment"})
        self.assertEqual(Path(p._resolve_weights()).name, "yolo11n-seg.pt")
        p = make_plugin(self.dir, {"task": "pose"})
        self.assertEqual(Path(p._resolve_weights()).name, "yolo11n-pose.pt")

    def test_explizite_auswahl_hat_vorrang(self):
        p = make_plugin(self.dir, {"yolo_model": "yolo11x.pt"})
        self.assertEqual(Path(p._resolve_weights()).name, "yolo11x.pt")

    def test_ohne_gewichte_bleibt_der_download_fallback(self):
        empty = Path(tempfile.mkdtemp())
        p = make_plugin(empty)
        self.assertEqual(p._resolve_weights(), "yolov8n.pt")


class FakeTrainer:
    """Nachbau der Ultralytics-Trainer-Attribute, die der Callback liest."""

    def __init__(self, tloss=(1.5, 2.0, 1.0), metrics=None, lr=None):
        self.tloss = tloss
        self.metrics = metrics or {}
        self.lr = lr or {"lr/pg0": 0.00123}

    def label_loss_items(self, tloss, prefix="train"):
        keys = ["box_loss", "cls_loss", "dfl_loss"]
        return {f"{prefix}/{k}": float(v) for k, v in zip(keys, tloss)}


class MetricsTest(unittest.TestCase):
    """Regression: Train Loss stand im Trainingsdialog dauerhaft auf 0.0000.

    Der Callback las trainer.metrics – dort stehen aber nur die
    Validierungswerte, nicht die Trainings-Losses.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.plugin = make_plugin(self.tmp.name)

    def test_running_loss_summiert_box_cls_dfl(self):
        self.assertAlmostEqual(self.plugin._running_loss(FakeTrainer()), 4.5)

    def test_running_loss_faellt_auf_metrics_zurueck(self):
        t = FakeTrainer(tloss=None, metrics={"train/box_loss": 1.0, "train/cls_loss": 0.5})
        self.assertAlmostEqual(self.plugin._running_loss(t), 1.5)

    def test_val_loss_wird_getrennt_summiert(self):
        m = {"val/box_loss": 2.0, "val/cls_loss": 1.0, "metrics/mAP50(B)": 0.5}
        self.assertAlmostEqual(self.plugin._sum_prefixed(m, "val/"), 3.0)
        self.assertIsNone(self.plugin._sum_prefixed(m, "train/"))

    def test_lernrate_ist_immer_ein_float(self):
        self.assertAlmostEqual(self.plugin._current_lr(FakeTrainer()), 0.00123)
        # progress() ruft float() darauf auf – None waere ein Absturz.
        self.assertIsInstance(self.plugin._current_lr(object()), float)


class SplitSizesTest(unittest.TestCase):
    """Regression: Analyse zeigte "n_train 0 / n_val 0" trotz 463/116 Bildern."""

    def test_zaehlt_bilder_aus_der_dataset_yaml(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for split, n in (("train", 3), ("val", 2)):
            d = root / "images" / split
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{i}.jpg").write_bytes(b"x")
        (root / "dataset.yaml").write_text(
            f"path: {root}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  - 'a'\n",
            encoding="utf-8")
        p = make_plugin(root)
        p._yaml_path = str(root / "dataset.yaml")
        self.assertEqual(p._split_sizes(), {"n_train": 3, "n_val": 2})

    def test_ohne_yaml_keine_erfundenen_zahlen(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = make_plugin(tmp.name)
        self.assertEqual(p._split_sizes(), {"n_train": 0, "n_val": 0})


class StopTest(unittest.TestCase):
    """Regression: "Stoppen" blieb wirkungslos, das Training lief zu Ende.

    Die Engine ruft im Signal-Handler plugin.stop(). YOLOPlugin erbt nicht von
    TrainPlugin, wo die Methode definiert ist – der Handler lief in einen
    AttributeError.
    """

    def test_stop_setzt_das_flag(self):
        p = make_plugin(tempfile.mkdtemp())
        self.assertFalse(p.is_stopped)
        p.stop()
        self.assertTrue(p.is_stopped)

    def test_alle_eigenstaendigen_plugins_haben_stop(self):
        import importlib
        for mod in ("plugins.yolo.plugin", "plugins.canvas.plugin",
                    "plugins.image_classification.plugin"):
            m = importlib.import_module(mod)
            classes = [c for c in vars(m).values()
                       if isinstance(c, type) and c.__module__ == mod
                       and hasattr(c, "is_stopped") or (isinstance(c, type) and c.__module__ == mod and "Plugin" in c.__name__)]
            self.assertTrue(classes, f"keine Plugin-Klasse in {mod}")
            for c in classes:
                self.assertTrue(callable(getattr(c, "stop", None)),
                                f"{mod}.{c.__name__} hat kein stop()")


if __name__ == "__main__":
    unittest.main(verbosity=2)
