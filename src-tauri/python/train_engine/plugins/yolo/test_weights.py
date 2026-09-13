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

    def test_leere_version_nimmt_vorgaenger_statt_download(self):
        with tempfile.TemporaryDirectory() as d:
            versions = Path(d) / "versions"
            (versions / "ver_alt").mkdir(parents=True)
            (versions / "ver_alt" / "model.pt").write_bytes(b"x")
            leer = versions / "ver_leer"
            (leer / "train").mkdir(parents=True)
            (leer / "train" / "args.yaml").write_text("x")
            self.assertEqual(Path(make_plugin(leer)._resolve_weights()), versions / "ver_alt" / "model.pt")

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


class LabelPruefungTest(unittest.TestCase):
    """Ein Detektions-Training ohne Labels laeuft sonst wirkungslos durch:
    Ultralytics behandelt jedes Bild als Hintergrund, box- und dfl-Loss sind
    konstant 0 und mAP ebenfalls."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.plugin = make_plugin(self.root)

    def write_yaml(self, train, val=None):
        y = self.root / "dataset.yaml"
        y.write_text(
            f"path: {self.root}\ntrain: {train}\nval: {val or train}\nnc: 1\nnames: ['object']\n",
            encoding="utf-8")
        return y

    def bilder(self, d: Path, n: int, labels_in: Path = None):
        d.mkdir(parents=True, exist_ok=True)
        if labels_in:
            labels_in.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"bild_{i}.jpg").write_bytes(b"x")
            if labels_in:
                (labels_in / f"bild_{i}.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")

    def test_bilder_ohne_labels_brechen_ab(self):
        # Genau der Praxisfall: der Unterordner 'images' wurde als Dataset
        # importiert, train/ und val/ enthalten nur .jpg-Dateien.
        self.bilder(self.root / "train", 5)
        self.bilder(self.root / "val", 2)
        y = self.write_yaml(self.root / "train", self.root / "val")
        self.assertFalse(self.plugin._verify_labels(y))

    def test_nested_layout_wird_akzeptiert(self):
        self.bilder(self.root / "images" / "train", 4, self.root / "labels" / "train")
        self.bilder(self.root / "images" / "val", 2, self.root / "labels" / "val")
        y = self.write_yaml(self.root / "images" / "train", self.root / "images" / "val")
        self.assertTrue(self.plugin._verify_labels(y))

    def test_labels_neben_den_bildern_werden_akzeptiert(self):
        d = self.root / "train"
        self.bilder(d, 3, d)
        y = self.write_yaml(d)
        self.assertTrue(self.plugin._verify_labels(y))

    def test_teilweise_labels_laufen_weiter(self):
        # Bilder ohne Label sind gueltige Hintergrundbilder - nur eine Warnung.
        d = self.root / "images" / "train"
        lab = self.root / "labels" / "train"
        self.bilder(d, 4, lab)
        for i in range(4, 10):
            (d / f"bild_{i}.jpg").write_bytes(b"x")
        y = self.write_yaml(d)
        self.assertTrue(self.plugin._verify_labels(y))

    def test_relative_pfade_in_der_yaml(self):
        self.bilder(self.root / "images" / "train", 3, self.root / "labels" / "train")
        y = self.root / "dataset.yaml"
        y.write_text(
            f"path: {self.root}\ntrain: images/train\nval: images/train\nnc: 1\nnames: ['object']\n",
            encoding="utf-8")
        self.assertEqual(len(self.plugin._yaml_image_dirs(y)), 2)
        self.assertTrue(self.plugin._verify_labels(y))

    def test_fehlende_split_ordner_brechen_mit_klarer_meldung_ab(self):
        # Die alte App-Split-yaml: images/train eingetragen, Dateien in train/images.
        self.bilder(self.root / "train" / "images", 3, self.root / "train" / "labels")
        self.bilder(self.root / "val" / "images", 1, self.root / "val" / "labels")
        y = self.root / "dataset.yaml"
        y.write_text(f"path: {self.root}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['o']\n",
                     encoding="utf-8")
        self.assertFalse(self.plugin._verify_labels(y))

    def test_roboflow_pfade_mit_punkt_punkt_gelten_als_vorhanden(self):
        self.bilder(self.root / "train" / "images", 2, self.root / "train" / "labels")
        y = self.root / "data.yaml"
        y.write_text("train: ../train/images\nval: ../train/images\nnc: 1\nnames: ['o']\n",
                     encoding="utf-8")
        self.assertTrue(self.plugin._verify_labels(y))

    def test_yaml_aus_plugin_config_hat_vorrang(self):
        eigene = self.root / "sub" / "eigene.yaml"
        eigene.parent.mkdir()
        eigene.write_text("train: x\n", encoding="utf-8")
        (self.root / "dataset.yaml").write_text("train: y\n", encoding="utf-8")
        self.plugin.config.plugin_config = {"dataset_yaml_path": str(eigene)}
        self.assertEqual(self.plugin._find_or_build_yaml(self.root), eigene)


if __name__ == "__main__":
    unittest.main(verbosity=2)
