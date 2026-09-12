"""Canvas-Loader: Split-Dateien, Text-Labels, ungelabelte Zeilen, Bildordner.

Aufruf:  python3 -m plugins.canvas.test_dataloaders   (aus train_engine/)
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import core  # noqa: F401  (macht ft_data importierbar)
from plugins.canvas.dataloaders import get_dataloaders


def ir(data_type, num_classes=2, task_type="classification", **params):
    return NS(training=NS(num_classes=num_classes, task_type=task_type),
              data=NS(type=data_type, params=params), execution_order=[], node_by_id=lambda: {})


class CanvasLoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def frame(self, labels):
        import numpy as np
        import pandas as pd
        n = len(labels)
        return pd.DataFrame({"f1": np.arange(n, dtype=float), "f2": np.ones(n), "name": ["x"] * n, "label": labels})

    def test_csv_train_val_test_im_root_und_text_labels(self):
        self.frame(["cat", "dog"] * 10).to_csv(self.root / "train.csv", index=False)
        self.frame(["cat", "dog"]).to_csv(self.root / "val.csv", index=False)
        self.frame([-1] * 7).to_csv(self.root / "test.csv", index=False)
        tr, va = get_dataloaders(ir("csv_loader"), str(self.root), 4)
        self.assertEqual((len(tr.dataset), len(va.dataset)), (20, 2))
        self.assertEqual(set(int(y) for _, y in tr.dataset), {0, 1})

    def test_labels_ab_eins_und_minus_eins(self):
        (self.root / "train").mkdir()
        self.frame([1, 2, 1, 2, -1]).to_parquet(self.root / "train" / "a.parquet")
        tr, va = get_dataloaders(ir("parquet_loader"), str(self.root), 4)
        ys = {int(y) for _, y in tr.dataset} | {int(y) for _, y in va.dataset}
        self.assertEqual(ys, {0, 1})
        self.assertEqual(len(tr.dataset) + len(va.dataset), 4)

    def test_zu_viele_klassen_fuer_das_modell(self):
        self.frame(["a", "b", "c"] * 3).to_csv(self.root / "train.csv", index=False)
        with self.assertRaises(ValueError):
            get_dataloaders(ir("csv_loader", num_classes=2), str(self.root), 4)

    def test_regression_behaelt_zielwerte(self):
        self.frame([-1.5, 2.0, 3.0, 4.0, 5.0]).to_csv(self.root / "train.csv", index=False)
        tr, va = get_dataloaders(ir("csv_loader", num_classes=1, task_type="regression"), str(self.root), 4)
        ys = sorted(float(y) for _, y in tr.dataset) + sorted(float(y) for _, y in va.dataset)
        self.assertIn(-1.5, ys)

    def test_bildordner_train_test_ohne_val(self):
        from PIL import Image
        for split in ("train", "test"):
            for cls in ("a", "b"):
                d = self.root / split / cls
                d.mkdir(parents=True)
                for i in range(3):
                    Image.new("RGB", (8, 8)).save(d / f"{i}.png")
        tr, va = get_dataloaders(ir("image_loader", imageSize=8), str(self.root), 2)
        self.assertEqual(len(tr.dataset) + len(va.dataset), 6)  # test/ bleibt draussen


if __name__ == "__main__":
    unittest.main(verbosity=2)
