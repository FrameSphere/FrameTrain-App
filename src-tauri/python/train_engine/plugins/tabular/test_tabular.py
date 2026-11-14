"""Tabellen-Plugin: Laden, fehlende Werte, Aufgabenerkennung, Kennzahlen, Mini-Training.

Aufruf:  python3.11 -m unittest plugins.tabular.test_tabular   (aus train_engine/)
Nur CPU, winzige Daten — laeuft in wenigen Sekunden.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import tabular_core as tc  # noqa: E402


def make_frame(n=400, seed=0, missing=0.1):
    """Klasse haengt von Zahl und Kategorie ab; Regressionsziel ebenfalls."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(0, 100, n)
    city = rng.choice(["Berlin", "Mainz", "Koeln"], n)
    label = np.where(x1 + (city == "Mainz") * 1.5 > 0.5, "ja", "nein")
    price = 3 * x1 + 0.05 * x2 + (city == "Berlin") * 4 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"kunden_id": [f"k{i}" for i in range(n)], "x1": x1, "x2": x2, "city": city,
                       "label": label, "price": price})
    m = rng.random(n) < missing
    df.loc[m, "x2"] = np.nan
    df.loc[rng.random(n) < missing, "city"] = None
    return df


class LoadAndPreprocessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_split_ordner_und_dateinamen_werden_erkannt(self):
        for d in ("train", "val", "test"):
            (self.root / d).mkdir()
            make_frame(30).to_csv(self.root / d / "data.csv", index=False)
        frames = tc.load_frames(str(self.root))
        self.assertEqual(set(frames), {"train", "val", "test"})
        other = self.root / "flat"
        other.mkdir()
        make_frame(30).to_parquet(other / "kunden_train.parquet")
        make_frame(10).to_csv(other / "kunden_test.csv", sep=";", index=False)
        frames = tc.load_frames(str(other))
        self.assertEqual((len(frames["train"]), len(frames["test"])), (30, 10))
        self.assertIn("x1", frames["test"].columns)  # Semikolon erkannt

    def test_fehlende_werte_imputation_und_indikator(self):
        df = make_frame(200, missing=0.2)
        prep = tc.Preprocessor().fit(df, "label")
        self.assertEqual(prep.task, "classification")
        self.assertIn("kunden_id", prep.dropped)  # ID-Spalte ignoriert
        self.assertEqual([c["name"] for c in prep.numeric], ["x1", "x2", "price"])
        x_num, x_cat = prep.transform(df)
        self.assertFalse(np.isnan(x_num).any())
        # x2 hat Luecken -> zusaetzliche Indikatorspalte
        self.assertEqual(x_num.shape[1], 4)
        miss_col = x_num[:, 3]
        np.testing.assert_array_equal(miss_col, df["x2"].isna().to_numpy().astype(float))
        # fehlende Kategorie -> Index 0, unbekannte Kategorie im Test -> 0
        self.assertTrue((x_cat[df["city"].isna().to_numpy(), 0] == 0).all())
        _, xc = prep.transform(pd.DataFrame({"x1": [0.0], "x2": [1.0], "city": ["Paris"], "price": [1.0]}))
        self.assertEqual(int(xc[0, 0]), 0)
        # Spalte fehlt ganz -> Median, kein Absturz
        xn, _ = prep.transform(pd.DataFrame({"city": ["Mainz"]}))
        self.assertEqual(xn.shape, (1, 4))

    def test_aufgabe_wird_erkannt(self):
        self.assertEqual(tc.infer_task(pd.Series(["a", "b", "a"])), "classification")
        self.assertEqual(tc.infer_task(pd.Series([0, 1, 1, 0] * 10)), "classification")
        self.assertEqual(tc.infer_task(pd.Series(np.linspace(0, 1, 100))), "regression")
        self.assertEqual(tc.infer_task(pd.Series(range(200))), "regression")
        self.assertEqual(tc.infer_task(pd.Series([0, 1]), "regression"), "regression")

    def test_vorverarbeitung_rundreise_json(self):
        df = make_frame(100)
        prep = tc.Preprocessor().fit(df, "price")
        self.assertEqual(prep.task, "regression")
        again = tc.Preprocessor.from_dict(json.loads(json.dumps(prep.to_dict())))
        a, b = prep.transform(df), again.transform(df)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_zufallsaufteilung_deterministisch_und_disjunkt(self):
        tr, va, te = tc.random_split(100, 0.15, 0.15, 42)
        tr2, _, te2 = tc.random_split(100, 0.15, 0.15, 42)
        np.testing.assert_array_equal(te, te2)
        np.testing.assert_array_equal(tr, tr2)
        self.assertEqual(len(set(tr) | set(va) | set(te)), 100)
        self.assertEqual(len(set(tr) & set(te)), 0)

    def test_zeile_parsen(self):
        cols = ["x1", "x2", "city"]
        self.assertEqual(tc.parse_row('{"x1": 1, "city": "Mainz"}', cols), {"x1": 1, "city": "Mainz"})
        self.assertEqual(tc.parse_row("x1=1.5; city=Mainz", cols), {"x1": 1.5, "city": "Mainz"})
        self.assertEqual(tc.parse_row("1.5, 20, Berlin", cols), {"x1": 1.5, "x2": 20.0, "city": "Berlin"})
        self.assertIsNone(tc.parse_row("1.5, ?, Berlin", cols)["x2"])
        with self.assertRaises(ValueError):
            tc.parse_row("1, 2", cols)


class MetricsTest(unittest.TestCase):
    def test_klassifikation(self):
        m = tc.classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2)
        self.assertAlmostEqual(m["accuracy"], 0.75)
        # F1 Klasse 0 = 2/3, Klasse 1 = 0.8 -> Mittel 0.7333
        self.assertAlmostEqual(m["macro_f1"], (2 / 3 + 0.8) / 2)

    def test_regression(self):
        m = tc.regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
        self.assertAlmostEqual(m["mae"], 2 / 3)
        self.assertAlmostEqual(m["rmse"], np.sqrt(4 / 3))
        self.assertAlmostEqual(m["r2"], 1 - 4 / 2)
        self.assertAlmostEqual(tc.regression_metrics(np.array([1.0, 3.0]), np.array([1.0, 3.0]))["r2"], 1.0)


class MiniTrainingTest(unittest.TestCase):
    """Plugin-Klasse direkt, ohne Subprozess: train -> export -> laden -> vorhersagen."""

    def run_plugin(self, target, task="auto", epochs=40):
        sys.path.insert(0, str(HERE.parent.parent))
        from core.config import TrainingConfig
        import importlib.util
        spec = importlib.util.spec_from_file_location("tabular_plugin_test", HERE / "plugin.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        make_frame(500, seed=1).to_csv(root / "daten.csv", index=False)
        cfg = TrainingConfig(dataset_path=str(root / "daten.csv"), output_path=str(root / "out"),
                             task_type="tabular", epochs=epochs, batch_size=64, learning_rate=3e-3,
                             weight_decay=0.0, plugin_config={"target_column": target, "task": task, "patience": 0})
        p = mod.Plugin(cfg)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            p.setup(); p.load_data(); p.build_model(); p.train()
            res = p.validate(); out = p.export()
        return res, out

    def test_klassifikation_schlaegt_haeufigste_klasse(self):
        res, out = self.run_plugin("label")
        self.assertEqual(res["task"], "classification")
        self.assertGreater(res["test_accuracy"], res["test_baseline_accuracy"] + 0.15)
        self.assertGreater(res["test_accuracy"], 0.85)
        model, prep = tc.load_bundle(out)
        pred = tc.predict_frame(model, prep, pd.DataFrame([{"x1": 2.0, "x2": 10, "city": "Mainz"},
                                                           {"x1": -2.0, "x2": 10, "city": "Berlin"}]))
        self.assertEqual([prep.classes[i] for i in pred["pred"]], ["ja", "nein"])

    def test_regression_schlaegt_mittelwert(self):
        res, out = self.run_plugin("price", epochs=60)
        self.assertEqual(res["task"], "regression")
        self.assertLess(res["test_mae"], 0.5 * res["test_baseline_mae"])
        self.assertGreater(res["test_r2"], 0.8)
        self.assertTrue((Path(out) / "preprocessing.json").is_file())
        self.assertEqual(json.loads((Path(out) / "model.json").read_text())["task"], "tabular")


if __name__ == "__main__":
    unittest.main()
