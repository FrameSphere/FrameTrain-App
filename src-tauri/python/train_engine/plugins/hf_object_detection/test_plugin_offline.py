"""HF-Detektor-Plugins ohne Netz: winziges Zufalls-YOLOS, 6 Bilder, 1 Epoche.

Prueft den ganzen Weg Train-Plugin -> gespeicherte Version -> Test-Plugin
(Einzelbild und Dataset) samt Ergebnisaufbau wie beim YOLO-Test (mAP50 zuerst,
detections mit x1..y2). Echte Laeufe mit hustvl/yolos-tiny und RT-DETR stehen
in docs/engine-ausbau/erkennung-detr.md.

Aufruf:  python3 plugins/hf_object_detection/test_plugin_offline.py   (aus train_engine/)
"""
import importlib.util, json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from core.config import TrainingConfig                 # noqa: E402
from plugins.hf_object_detection.plugin import Plugin  # noqa: E402

TEST_ENGINE = HERE.parents[3] / "test_engine"


def load_test_plugin():
    """Test-Plugin mit eigenem core-Paket der Test-Engine laden."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "core" or k.startswith("core.")}
    sys.path.insert(0, str(TEST_ENGINE))
    try:
        spec = importlib.util.spec_from_file_location(
            "ft_test_hf_od", TEST_ENGINE / "plugins" / "hf_object_detection" / "plugin.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from core.config import TestConfig
        return mod, TestConfig
    finally:
        sys.path.remove(str(TEST_ENGINE))
        for k in [k for k in sys.modules if k == "core" or k.startswith("core.")]:
            sys.modules.pop(k)
        sys.modules.update(saved)


def tiny_yolos(path: Path):
    from transformers import YolosConfig, YolosForObjectDetection, YolosImageProcessor
    cfg = YolosConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                      image_size=[64, 64], patch_size=16, num_detection_tokens=5, num_labels=3)
    YolosForObjectDetection(cfg).save_pretrained(path)
    YolosImageProcessor(size={"shortest_edge": 64, "longest_edge": 64}).save_pretrained(path)


def tiny_dataset(root: Path):
    from PIL import Image
    for split, n in (("train", 4), ("val", 1), ("test", 1)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (80, 60), (40 * i, 90, 160)).save(root / split / "images" / f"{i}.jpg")
            (root / split / "labels" / f"{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.1\n")
    (root / "data.yaml").write_text("names:\n  0: box\n  1: dot\n")


class OfflineRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.model_dir, cls.data, cls.out = base / "yolos", base / "data", base / "out"
        tiny_yolos(cls.model_dir)
        tiny_dataset(cls.data)
        cfg = TrainingConfig(model_path=str(cls.model_dir), dataset_path=str(cls.data),
                             output_path=str(cls.out), task_type="hf_object_detection",
                             epochs=1, batch_size=2, learning_rate=1e-4,
                             plugin_config={"imgsz": 64, "device": "cpu"})
        cls.messages = []
        with mock.patch("core.protocol.MessageProtocol._send",
                        side_effect=lambda t, d: cls.messages.append((t, d))):
            p = Plugin(cfg)
            assert p.setup() is not False and p.load_data() is not False and p.build_model() is not False
            assert p.train() is not False
            cls.metrics = {**p.validate(), **p.get_metrics()}
            p.export()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_training_meldet_fortschritt_und_map(self):
        progress = [d for t, d in self.messages if t == "progress"]
        self.assertTrue(progress)
        self.assertEqual(progress[-1]["total_steps"], 2)
        self.assertIn("mAP50", progress[-1]["metrics"])
        self.assertEqual(self.metrics["n_train"], 4)
        self.assertEqual(self.metrics["architecture"], "yolos")

    def test_export_hat_klassen_und_task(self):
        meta = json.loads((self.out / "model.json").read_text())
        self.assertEqual(meta["task"], "hf_object_detection")
        self.assertEqual(meta["classes"], ["box", "dot"])
        cfg = json.loads((self.out / "config.json").read_text())
        self.assertEqual(cfg["id2label"], {"0": "box", "1": "dot"})

    def test_test_plugin_dataset_und_einzelbild(self):
        mod, TestConfig = load_test_plugin()
        sent = []
        out = self.out.parent / "test_out"
        with mock.patch.object(mod.TestProtocol, "_send", side_effect=lambda t, d: sent.append((t, d))):
            cfg = TestConfig.from_dict({"model_path": str(self.out), "dataset_path": str(self.data),
                                        "output_path": str(out), "task_type": "hf_object_detection",
                                        "batch_size": 2, "plugin_config": {"device": "cpu", "conf": 0.0}})
            plugin = mod.Plugin(cfg)
            plugin.setup()
            plugin.run_dataset()
            done = [d for t, d in sent if t == "complete"][-1]
            self.assertEqual(list(done["metrics"])[:2], ["mAP50", "mAP50-95"])
            self.assertIsNone(done["accuracy"])
            rows = json.loads(Path(done["results_file"]).read_text())["predictions"]
            self.assertEqual(len(rows), 1)  # test/ hat Vorrang vor val/
            self.assertEqual(rows[0]["expected_output"], "1x box, 1x dot")
            for d in rows[0]["detections"]:
                self.assertTrue({"label", "confidence", "x1", "y1", "x2", "y2"} <= set(d))

            plugin.config.single_input = str(self.data / "test" / "images" / "0.jpg")
            plugin.run_single()
            single = [d for t, d in sent if t == "complete"][-1]
            self.assertEqual(single["mode"], "single")
            self.assertEqual((single["image_width"], single["image_height"]), (80, 60))
            self.assertIsInstance(single["detections"], list)

    def test_ultralytics_ordner_wird_abgelehnt(self):
        from ft_data.hf_detection import check_model_dir
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "rtdetr-l.pt").write_bytes(b"x")
            self.assertIn("YOLO-Plugin", check_model_dir(Path(d)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
