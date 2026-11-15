"""Ende-zu-Ende-Test: winziger SegFormer durch Train- und Test-Engine.

Das Modell wird lokal aus einer Mini-Config erzeugt (rund 9.000 Parameter, kein
Download). Der Datensatz: 8 farbige Rauschbilder mit einem hellen Quadrat
(Klasse 1) auf Hintergrund (Klasse 0), dazu ein test/-Split. Geprueft wird der
ganze Weg wie in der App: train_engine.py --config, JSON-Nachrichten auf
stdout, gespeichertes Modell mit label_mapping.json, danach test_engine.py im
Dataset- und Einzelbild-Modus.

Aufruf:  python3.11 -m unittest plugins.semantic_segmentation.test_plugin   (aus train_engine/)
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TRAIN_ENGINE = Path(__file__).resolve().parents[2]
TEST_ENGINE = TRAIN_ENGINE.parent / "test_engine"


def make_dataset(root: Path, color_masks: bool):
    from PIL import Image
    rng = np.random.default_rng(0)
    pal = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)
    for split, n in (("train", 8), ("val", 2), ("test", 3)):
        for i in range(n):
            img = (rng.random((64, 64, 3)) * 80).astype(np.uint8)
            m = np.zeros((64, 64), dtype=np.uint8)
            y0, x0 = rng.integers(0, 36, size=2)
            m[y0:y0 + 28, x0:x0 + 28] = 1
            img[m == 1] = 230
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "masks").mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(root / split / "images" / f"{split}{i}.png")
            Image.fromarray(pal[m] if color_masks else m, mode=None).save(root / split / "masks" / f"{split}{i}.png")
    if color_masks:
        (root / "classes.txt").write_text("hintergrund 0 0 0\nquadrat 255 0 0\n", encoding="utf-8")


def make_tiny_segformer(path: Path):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor
    cfg = SegformerConfig(num_channels=3, num_encoder_blocks=2, depths=[1, 1], sr_ratios=[2, 1],
                          hidden_sizes=[8, 16], num_attention_heads=[1, 1], decoder_hidden_size=16,
                          patch_sizes=[7, 3], strides=[4, 2], num_labels=2)
    SegformerForSemanticSegmentation(cfg).save_pretrained(path)
    SegformerImageProcessor(size={"height": 64, "width": 64}).save_pretrained(path)


def run(script: Path, cfg: dict, tmp: Path):
    cfg_file = tmp / f"{script.stem}_{cfg.get('mode', 'train')}.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script), "--config", str(cfg_file)],
                          cwd=str(script.parent), capture_output=True, text=True, timeout=600)
    msgs = []
    for line in proc.stdout.splitlines():
        try:
            msgs.append(json.loads(line))
        except ValueError:
            pass
    return msgs, proc


class EngineRunTest(unittest.TestCase):
    def _run_case(self, color_masks: bool):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            data, model, out = tmp / "data", tmp / "model", tmp / "out"
            make_dataset(data, color_masks)
            make_tiny_segformer(model)
            msgs, proc = run(TRAIN_ENGINE / "train_engine.py", {
                "task_type": "semantic_segmentation", "model_path": str(model),
                "dataset_path": str(data), "output_path": str(out), "checkpoint_dir": str(tmp / "ckpt"),
                "epochs": 3, "batch_size": 4, "learning_rate": 5e-3, "warmup_ratio": 0.0,
                "logging_steps": 1, "eval_strategy": "epoch", "save_steps": 1000,
            }, tmp)
            errors = [m for m in msgs if m["type"] == "error"]
            self.assertFalse(errors, f"{errors}\n{proc.stderr[-3000:]}")
            done = [m for m in msgs if m["type"] == "complete"]
            self.assertEqual(len(done), 1, proc.stderr[-3000:])
            fm = done[0]["data"]["final_metrics"]
            for k in ("mean_iou", "pixel_accuracy", "final_val_loss"):
                self.assertIn(k, fm)
            self.assertGreaterEqual(fm["mean_iou"], 0.0)
            progress_metrics = [m["data"]["metrics"] for m in msgs if m["type"] == "progress" and m["data"]["metrics"]]
            self.assertTrue(any("eval_mean_iou" in pm for pm in progress_metrics), "mIoU nicht im Fortschritt")
            mapping = json.loads((out / "label_mapping.json").read_text(encoding="utf-8"))
            self.assertEqual(mapping["mask_mode"], "color" if color_masks else "index")
            if color_masks:
                self.assertEqual(mapping["classes"], ["hintergrund", "quadrat"])

            tmsgs, tproc = run(TEST_ENGINE / "test_engine.py", {
                "task_type": "semantic_segmentation", "model_path": str(out),
                "dataset_path": str(data), "output_path": str(tmp / "test_out"), "mode": "dataset",
            }, tmp)
            tdone = [m for m in tmsgs if m["type"] == "complete"]
            self.assertEqual(len(tdone), 1, f"{tmsgs[-3:]}\n{tproc.stderr[-3000:]}")
            td = tdone[0]["data"]
            self.assertEqual(td["total_samples"], 3)
            self.assertEqual(td["split"], "test")
            self.assertIn("mean_iou", td["metrics"])
            self.assertEqual(len(td["per_class"]), 2)
            self.assertTrue(all(Path(p).is_file() for p in td["images"]))

            img = data / "test" / "images" / "test0.png"
            smsgs, sproc = run(TEST_ENGINE / "test_engine.py", {
                "task_type": "semantic_segmentation", "model_path": str(out),
                "output_path": str(tmp / "single"), "mode": "single",
                "single_input": str(img), "single_input_type": "file",
            }, tmp)
            sdone = [m for m in smsgs if m["type"] == "complete"]
            self.assertEqual(len(sdone), 1, f"{smsgs[-3:]}\n{sproc.stderr[-3000:]}")
            sd = sdone[0]["data"]
            self.assertEqual(sd["output_kind"], "segmentation")
            for k in ("overlay_path", "color_mask_path", "mask_path"):
                self.assertTrue(Path(sd[k]).is_file(), k)
            self.assertAlmostEqual(sum(r["share"] for r in sd["legend"]), 1.0, places=5)
            return fm, td

    def test_klassen_id_masken(self):
        fm, td = self._run_case(color_masks=False)
        print(f"\n[ID-Masken] Train mIoU {fm['mean_iou']:.3f} | Test mIoU {td['metrics']['mean_iou']:.3f} "
              f"| Pixel-Acc {td['metrics']['pixel_accuracy']:.3f}")

    def test_farbmasken_mit_farbtabelle(self):
        fm, td = self._run_case(color_masks=True)
        print(f"\n[Farbmasken] Train mIoU {fm['mean_iou']:.3f} | Test mIoU {td['metrics']['mean_iou']:.3f} "
              f"| Pixel-Acc {td['metrics']['pixel_accuracy']:.3f}")


if __name__ == "__main__":
    unittest.main()
