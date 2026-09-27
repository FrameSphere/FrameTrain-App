"""Prueft die YOLO-Aufgaben segment, pose, obb und classify ohne Ultralytics-Lauf.

Hintergrund: Das Plugin konnte nur Boxen. Seg-, Pose-, OBB- und Cls-Gewichte
lagen im Modellordner, aber Label-Pruefung, dataset.yaml und Metriken kannten
nur detect; ein Klassifikations-Dataset (Ordner pro Klasse) scheiterte schon an
"keine Labels".

Aufruf:  python3 plugins/yolo/test_tasks.py   (aus train_engine/)
"""
import sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.config import TrainingConfig              # noqa: E402
from core.plugin_base import TrainPlugin            # noqa: E402
from plugins.yolo.plugin import (                   # noqa: E402
    YOLOPlugin, guess_task_from_labels, infer_kpt_shape, is_class_folder_dataset,
    normalize_task, resume_requested, task_from_weights_name, task_metrics,
)

POSE_LINE = "0 " + " ".join(["0.5"] * 4) + " " + " ".join(["0.1 0.2 2"] * 17)   # 56 Werte
SEG_LINES = ["3 0.1 0.1 0.2 0.1 0.2 0.3", "1 0.1 0.1 0.2 0.1 0.2 0.3 0.1 0.3 0.05 0.2"]
OBB_LINE = "1 0.81 0.52 0.81 0.50 0.87 0.50 0.87 0.52"


def make_plugin(tmp: Path, plugin_config=None, model_dir=None) -> YOLOPlugin:
    cfg = TrainingConfig()
    cfg.model_path = str(model_dir or tmp)
    cfg.dataset_path = str(tmp)
    cfg.output_path = str(tmp / "out" / "final_model")
    cfg.checkpoint_dir = str(tmp / "out" / "checkpoints")
    cfg.plugin_config = plugin_config or {}
    p = YOLOPlugin(cfg)
    p._output_dir = Path(cfg.output_path)
    return p


def yolo_dataset(root: Path, lines, n: int = 3) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (root / "images" / split / f"b{i}.jpg").write_bytes(b"x")
            (root / "labels" / split / f"b{i}.txt").write_text("\n".join(lines), encoding="utf-8")
    y = root / "dataset.yaml"
    y.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnc: 4\nnames: ['a','b','c','d']\n",
                 encoding="utf-8")
    return y


class ReineLogikTest(unittest.TestCase):
    def test_aufgabennamen(self):
        self.assertEqual(normalize_task(None), "auto")
        self.assertEqual(normalize_task("Seg"), "segment")
        self.assertEqual(normalize_task("cls"), "classify")
        self.assertEqual(normalize_task("keypoints"), "pose")
        self.assertEqual(normalize_task("quatsch"), "auto")

    def test_aufgabe_aus_gewichtsnamen(self):
        self.assertEqual(task_from_weights_name("yolo11n.pt"), "detect")
        self.assertEqual(task_from_weights_name("/x/yolo11n-seg.pt"), "segment")
        self.assertEqual(task_from_weights_name("yolov8m-pose.pt"), "pose")
        self.assertEqual(task_from_weights_name("yolo11n-obb.pt"), "obb")
        self.assertEqual(task_from_weights_name("yolo11n-cls.pt"), "classify")
        self.assertIsNone(task_from_weights_name("model.pt"))

    def test_kpt_shape_aus_der_labelbreite(self):
        self.assertEqual(infer_kpt_shape([56, 56]), [17, 3])
        self.assertEqual(infer_kpt_shape([13]), [4, 2], "8 Werte gehen nur als 4 Punkte (x, y) auf")
        self.assertEqual(infer_kpt_shape([11]), [2, 3])
        self.assertIsNone(infer_kpt_shape([56, 20]), "unterschiedlich breite Zeilen sind keine Pose")
        self.assertIsNone(infer_kpt_shape([5, 5]))

    def test_aufgabe_aus_labelzeilen(self):
        self.assertEqual(guess_task_from_labels([5, 5]), "detect")
        self.assertEqual(guess_task_from_labels([9, 9]), "obb")
        self.assertEqual(guess_task_from_labels([56, 56]), "pose")
        self.assertEqual(guess_task_from_labels([7, 11, 25]), "segment")
        self.assertIsNone(guess_task_from_labels([]))

    def test_metriken_je_aufgabe(self):
        m = {"metrics/mAP50(B)": 0.8, "metrics/mAP50-95(B)": 0.6, "metrics/precision(B)": 0.7,
             "metrics/recall(B)": 0.9, "metrics/mAP50(M)": 0.5, "metrics/mAP50-95(M)": 0.3,
             "metrics/mAP50(P)": 0.4, "metrics/mAP50-95(P)": 0.2}
        det = task_metrics("detect", m)
        self.assertEqual(set(det), {"mAP50", "mAP50-95", "precision", "recall"},
                         "Detect-Schluessel bleiben unveraendert")
        seg = task_metrics("segment", m)
        self.assertEqual(seg["mAP50"], 0.8)
        self.assertEqual(seg["mask_mAP50"], 0.5)
        self.assertEqual(seg["mask_mAP50-95"], 0.3)
        self.assertEqual(task_metrics("pose", m)["pose_mAP50"], 0.4)
        self.assertEqual(set(task_metrics("obb", m)), set(det))
        cls = task_metrics("classify", {"metrics/accuracy_top1": 0.75, "metrics/accuracy_top5": 1.0})
        self.assertEqual(cls, {"accuracy": 0.75, "top1_accuracy": 0.75, "top5_accuracy": 1.0})

    def test_resume_schalter(self):
        for v in (True, "auto", "true", "/pfad/last.pt"):
            self.assertTrue(resume_requested(v), v)
        for v in (False, "", None, "false", "0"):
            self.assertFalse(resume_requested(v), v)

    def test_klassifikation_loss_heisst_nur_loss(self):
        self.assertAlmostEqual(YOLOPlugin._sum_prefixed({"val/loss": 0.7, "metrics/accuracy_top1": 1}, "val/"), 0.7)


class ErbtVonTrainPluginTest(unittest.TestCase):
    def test_ist_ein_trainplugin(self):
        self.assertTrue(issubclass(YOLOPlugin, TrainPlugin))
        with tempfile.TemporaryDirectory() as d:
            p = make_plugin(Path(d))
            self.assertFalse(p.is_stopped)
            self.assertEqual(p.task, "detect", "vor setup() gilt detect wie bisher")


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def klassen(self, base: Path, names=("katze", "hund"), n=5):
        for name in names:
            (base / name).mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (base / name / f"{name}{i}.jpg").write_bytes(b"x")

    def test_ordner_pro_klasse_wird_erkannt(self):
        self.klassen(self.root / "ds")
        self.assertTrue(is_class_folder_dataset(self.root / "ds"))
        yolo_dataset(self.root / "det", ["0 0.5 0.5 0.1 0.1"])
        self.assertFalse(is_class_folder_dataset(self.root / "det"))

    def test_klassifikation_ohne_split_wird_aufgeteilt(self):
        self.klassen(self.root / "ds")
        p = make_plugin(self.root / "ds")
        p.config.dataset_path = str(self.root / "ds")
        p._output_dir = self.root / "out" / "final_model"
        p.config.checkpoint_dir = str(self.root / "out" / "checkpoints")
        data = p._prepare_classify_data(self.root / "ds")
        self.assertEqual(data, self.root / "out" / "checkpoints" / "cls_data",
                         "nicht im Modellordner, der als Version kopiert wird")
        for split in ("train", "val"):
            self.assertEqual(sorted(d.name for d in (data / split).iterdir()), ["hund", "katze"])
        self.assertEqual(len(list((data / "val" / "katze").iterdir())), 1)
        self.assertEqual(len(list((data / "train" / "katze").iterdir())), 4)

    def test_fertiger_split_bleibt_wie_er_ist(self):
        self.klassen(self.root / "ds" / "train")
        self.klassen(self.root / "ds" / "val", n=2)
        p = make_plugin(self.root / "ds")
        self.assertEqual(p._prepare_classify_data(self.root / "ds"), self.root / "ds")

    def test_eine_klasse_reicht_nicht(self):
        self.klassen(self.root / "ds", names=("nur",))
        self.assertIsNone(make_plugin(self.root)._prepare_classify_data(self.root / "ds"))

    def test_segmentierung_braucht_polygone(self):
        p = make_plugin(self.root, {"task": "segment"})
        self.assertEqual(p.task, "segment")
        self.assertTrue(p._verify_labels(yolo_dataset(self.root / "seg", SEG_LINES)))
        self.assertFalse(p._verify_labels(yolo_dataset(self.root / "box", ["0 0.5 0.5 0.1 0.1"])),
                         "reine Box-Labels waeren ein wirkungsloses Masken-Training")

    def test_obb_braucht_vier_ecken(self):
        p = make_plugin(self.root, {"task": "obb"})
        self.assertTrue(p._verify_labels(yolo_dataset(self.root / "obb", [OBB_LINE])))
        self.assertFalse(p._verify_labels(yolo_dataset(self.root / "box", ["0 0.5 0.5 0.1 0.1"])))

    def test_pose_kpt_shape_wird_abgeleitet(self):
        y = yolo_dataset(self.root / "pose", [POSE_LINE])
        p = make_plugin(self.root, {"task": "pose"})
        self.assertTrue(p._verify_labels(y))
        derived = p._ensure_kpt_shape(y)
        self.assertNotEqual(derived, y, "das Dataset selbst bleibt unberuehrt")
        data = p._read_yaml(derived)
        self.assertEqual(data["kpt_shape"], [17, 3])
        self.assertEqual(Path(data["path"]), self.root / "pose")
        self.assertNotIn("kpt_shape", y.read_text())

    def test_pose_mit_kpt_shape_in_der_yaml(self):
        y = yolo_dataset(self.root / "pose", [POSE_LINE])
        y.write_text(y.read_text() + "kpt_shape: [17, 3]\n")
        p = make_plugin(self.root, {"task": "pose"})
        self.assertTrue(p._verify_labels(y))
        self.assertEqual(p._ensure_kpt_shape(y), y)

    def test_generierte_yaml_fuer_pose_hat_kpt_shape(self):
        ds = self.root / "pose"
        yolo_dataset(ds, [POSE_LINE])
        (ds / "dataset.yaml").unlink()
        p = make_plugin(ds, {"task": "pose"})
        y = p._generate_yaml(ds)
        self.assertIn("kpt_shape: [17, 3]", y.read_text())

    def test_klassen_bis_zur_hoechsten_id(self):
        # Vorher: IDs {0, 3} -> nc=2, Ultralytics brach mit "Label class 3 exceeds nc" ab.
        ds = self.root / "det"
        yolo_dataset(ds, ["0 0.5 0.5 0.1 0.1", "3 0.5 0.5 0.1 0.1"])
        self.assertEqual(make_plugin(ds)._detect_classes(ds), ["class_0", "class_1", "class_2", "class_3"])


class AufgabeWaehlenTest(unittest.TestCase):
    """Ein Basisordner wie Ultralytics/YOLO11 enthaelt mehrere Aufgaben."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.models = self.root / "models"
        self.models.mkdir()

    def weights(self, *names):
        for n in names:
            (self.models / n).write_bytes(b"x")

    def test_einzige_aufgabe_im_ordner_gewinnt(self):
        self.weights("yolo11n-seg.pt", "yolo11s-seg.pt")
        ds = self.root / "ds"
        yolo_dataset(ds, ["0 0.5 0.5 0.1 0.1"])
        self.assertEqual(make_plugin(ds, model_dir=self.models)._initial_task(ds), "segment")

    def test_gemischter_ordner_fragt_das_dataset(self):
        self.weights("yolo11n.pt", "yolo11n-seg.pt", "yolo11n-pose.pt")
        ds = self.root / "pose"
        yolo_dataset(ds, [POSE_LINE])
        p = make_plugin(ds, model_dir=self.models)
        p.task = p._initial_task(ds)
        self.assertEqual(p.task, "pose")
        self.assertEqual(Path(p._resolve_weights()).name, "yolo11n-pose.pt")

    def test_box_dataset_bleibt_bei_detect(self):
        self.weights("yolo11n.pt", "yolo11n-seg.pt", "yolo11n-pose.pt")
        ds = self.root / "det"
        yolo_dataset(ds, ["0 0.5 0.5 0.1 0.1"])
        p = make_plugin(ds, model_dir=self.models)
        self.assertEqual(p._initial_task(ds), "detect")

    def test_explizite_aufgabe_hat_vorrang(self):
        self.weights("yolo11n.pt", "yolo11n-seg.pt")
        ds = self.root / "det"
        yolo_dataset(ds, ["0 0.5 0.5 0.1 0.1"])
        self.assertEqual(make_plugin(ds, {"task": "seg"}, model_dir=self.models)._initial_task(ds), "segment")

    def test_passende_gewichte_werden_nicht_abgelehnt(self):
        self.weights("yolo11n-seg.pt")
        p = make_plugin(self.root, {"task": "segment"}, model_dir=self.models)
        p.yolo_model = str(self.models / "yolo11n-seg.pt")
        # Die Datei ist kein echter Checkpoint -> Name entscheidet: passt.
        self.assertTrue(p._settle_task())

    def test_falsche_gewichte_geben_klare_meldung(self):
        self.weights("yolo11n.pt")
        p = make_plugin(self.root, {"task": "segment"}, model_dir=self.models)
        p.yolo_model = str(self.models / "yolo11n.pt")
        self.assertFalse(p._settle_task())

    def test_auto_folgt_den_gewichten(self):
        self.weights("yolo11n-cls.pt")
        p = make_plugin(self.root, model_dir=self.models)
        p.yolo_model = str(self.models / "yolo11n-cls.pt")
        self.assertTrue(p._settle_task())
        self.assertEqual(p.task, "classify")


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_pfad_zum_alten_job_ordner(self):
        last = self.root / "job" / "final_model" / "train" / "weights" / "last.pt"
        last.parent.mkdir(parents=True)
        last.write_bytes(b"x")
        p = make_plugin(self.root, {"resume": str(self.root / "job")})
        self.assertEqual(p._resume_checkpoint(), last)

    def test_auto_sucht_in_der_version(self):
        version = self.root / "ver"
        (version / "train" / "weights").mkdir(parents=True)
        (version / "train" / "weights" / "last.pt").write_bytes(b"x")
        p = make_plugin(self.root, {"resume": "auto"}, model_dir=version)
        self.assertEqual(p._resume_checkpoint(), version / "train" / "weights" / "last.pt")

    def test_ohne_checkpoint_neu_starten(self):
        p = make_plugin(self.root, {"resume": True})
        self.assertIsNone(p._resume_checkpoint())


if __name__ == "__main__":
    unittest.main(verbosity=2)
