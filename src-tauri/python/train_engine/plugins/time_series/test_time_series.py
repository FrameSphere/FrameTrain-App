"""Zeitreihen-Plugin: Laden, Luecken, zeitliche Aufteilung ohne Leck, Kennzahlen, Mini-Training.

Aufruf:  python3.11 -m unittest plugins.time_series.test_time_series   (aus train_engine/)
Nur CPU, winziger PatchTST, wenige Epochen.
"""
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import ts_core as ts  # noqa: E402


def make_series_frame(n=300, ids=("a", "b"), period=12, seed=0, missing=0.0, shuffle=False):
    """Saisonale Reihen mit Trend und unterschiedlichem Niveau je Serie."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, sid in enumerate(ids):
        t = np.arange(n)
        y = 10 * (k + 1) + 0.02 * t + 3 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.2, n)
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        rows.append(pd.DataFrame({"timestamp": dates.astype(str), "store": sid, "sales": y}))
    df = pd.concat(rows, ignore_index=True)
    if missing:
        df.loc[rng.random(len(df)) < missing, "sales"] = np.nan
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


class LoadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_spalten_automatisch_erkannt(self):
        df = make_series_frame(50)
        cols = ts.detect_columns(df)
        self.assertEqual((cols.time, cols.values, cols.id), ("timestamp", ["sales"], "store"))
        cols = ts.detect_columns(df.rename(columns={"timestamp": "wann"}))
        self.assertEqual(cols.time, "wann")  # am Inhalt als Datum erkannt
        with self.assertRaises(ValueError):
            ts.detect_columns(df, target_columns="gibtsnicht")

    def test_zufaellig_aufgeteilte_dateien_werden_zeitlich_sortiert(self):
        # So sieht ein Dataset nach dem Split-Knopf der App aus: Zeilen gemischt.
        df = make_series_frame(120, shuffle=True)
        for i, d in enumerate(("train", "val", "test")):
            (self.root / d).mkdir()
            df.iloc[i::3].to_csv(self.root / d / "data.csv", index=False)
        series = ts.build_series(ts.load_all(str(self.root)), ts.detect_columns(df))
        ref = ts.build_series(make_series_frame(120), ts.detect_columns(df))
        self.assertEqual(set(series), {"a", "b"})
        np.testing.assert_allclose(series["a"], ref["a"], rtol=1e-6)

    def test_fehlende_werte_interpoliert(self):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=6, freq="D"),
                           "v": [1.0, np.nan, 3.0, np.nan, np.nan, 6.0]})
        arr = ts.build_series(df, ts.detect_columns(df))["series"]
        np.testing.assert_allclose(arr[:, 0], [1, 2, 3, 4, 5, 6])
        df.loc[0, "v"] = np.nan  # Rand
        arr = ts.build_series(df, ts.detect_columns(df))["series"]
        self.assertFalse(np.isnan(arr).any())

    def test_ohne_zeitspalte_gilt_zeilenfolge(self):
        df = pd.DataFrame({"a": [3.0, 1.0, 2.0]})
        cols = ts.detect_columns(df)
        self.assertIsNone(cols.time)
        np.testing.assert_allclose(ts.build_series(df, cols)["series"][:, 0], [3, 1, 2])


class SplitTest(unittest.TestCase):
    def test_kein_leck_zwischen_abschnitten(self):
        L, H = 24, 6
        sp = ts.split_points(200, H, 0.1, 0.2)
        self.assertEqual((sp.train_end, sp.val_end), (140, 160))
        train = ts.window_starts(sp, "train", L, H)
        val = ts.window_starts(sp, "val", L, H)
        test = ts.window_starts(sp, "test", L, H)
        # Trainingsziele enden vor dem Trainingsende
        self.assertLessEqual(max(s + L + H for s in train), sp.train_end)
        # Val-/Testziele liegen vollstaendig in ihrem Abschnitt
        for s in val:
            self.assertTrue(sp.train_end <= s + L and s + L + H <= sp.val_end)
        for s in test:
            self.assertTrue(sp.val_end <= s + L and s + L + H <= sp.length)
        # Testabschnitt lueckenlos in H-Bloecken abgedeckt
        self.assertEqual([s + L for s in test], list(range(160, 200 - H + 1, H)))

    def test_mindestens_ein_horizont_je_abschnitt(self):
        sp = ts.split_points(50, 10, 0.01, 0.01)
        self.assertEqual((sp.train_end, sp.val_end), (30, 40))

    def test_normalisierung_nur_aus_trainingsteil(self):
        arr = np.concatenate([np.zeros(80), np.full(20, 1000.0)])[:, None].astype(np.float32)
        norm = ts.Normalizer.fit({"s": arr}, {"s": 80})
        self.assertEqual(norm.mean["s"], [0.0])  # Testwerte (1000) fliessen nicht ein

    def test_fenster_inhalt_entspricht_der_reihe(self):
        arr = np.arange(100, dtype=np.float32)[:, None]
        norm = ts.Normalizer(mean={"s": [0.0]}, std={"s": [1.0]})
        splits = {"s": ts.split_points(100, 5, 0.1, 0.2)}
        past, fut, meta = ts.make_windows({"s": arr}, splits, norm, "test", 10, 5)
        s = meta[0][1]
        np.testing.assert_array_equal(past[0, :, 0], arr[s:s + 10, 0])
        np.testing.assert_array_equal(fut[0, :, 0], arr[s + 10:s + 15, 0])
        self.assertEqual(s + 10, splits["s"].val_end)


class MetricsTest(unittest.TestCase):
    def test_kennzahlen_von_hand(self):
        y = np.array([[[1.0], [2.0], [4.0]]])
        p = np.array([[[1.0], [3.0], [2.0]]])
        m = ts.forecast_metrics(y, p, scale=np.array([[2.0]]))
        self.assertAlmostEqual(m["mae"], 1.0)
        self.assertAlmostEqual(m["rmse"], np.sqrt(5 / 3))
        smape = (0 + 2 * 1 / 5 + 2 * 2 / 6) / 3 * 100
        self.assertAlmostEqual(m["smape"], smape)
        self.assertAlmostEqual(m["mase"], 0.5)

    def test_naive_vorhersagen(self):
        h = np.array([1, 2, 3, 4, 5, 6], dtype=float)[:, None]
        np.testing.assert_array_equal(ts.naive_forecast(h, 3)[:, 0], [6, 6, 6])
        np.testing.assert_array_equal(ts.naive_forecast(h, 5, season=3)[:, 0], [4, 5, 6, 4, 5])
        # MASE-Skala: mittlere Differenz zum Vorwert bzw. zur Vorsaison
        self.assertAlmostEqual(float(ts.mase_scale(h)[0]), 1.0)
        self.assertAlmostEqual(float(ts.mase_scale(h, season=3)[0]), 3.0)

    def test_eingabe_parsen(self):
        np.testing.assert_array_equal(ts.parse_history("1, 2 3;4", 1)[:, 0], [1, 2, 3, 4])
        self.assertEqual(ts.parse_history("1,2; 3,4", 2).shape, (2, 2))
        with self.assertRaises(ValueError):
            ts.parse_history("1,2; 3", 2)


class MiniTrainingTest(unittest.TestCase):
    def test_patchtst_neu_schlaegt_naiv(self):
        from core.config import TrainingConfig
        spec = importlib.util.spec_from_file_location("ts_plugin_test", HERE / "plugin.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        make_series_frame(400, ids=("a", "b", "c"), missing=0.03).to_csv(root / "sales.csv", index=False)
        cfg = TrainingConfig(dataset_path=str(root), output_path=str(root / "out"), task_type="time_series",
                             epochs=8, batch_size=32, learning_rate=3e-3, weight_decay=0.0,
                             plugin_config={"context_length": 48, "prediction_length": 12, "device": "cpu",
                                            "patience": 0, "d_model": 32, "ffn_dim": 64})
        p = mod.Plugin(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            p.setup(); p.load_data(); p.build_model(); p.train()
            res = p.validate(); out = p.export()
        self.assertEqual(res["eval_split"], "test")
        self.assertLess(res["test_mae"], res["test_naive_mae"])
        self.assertLess(res["test_mase"], 1.0)
        model, meta = ts.load_trained(out)
        self.assertEqual((meta["context_length"], meta["value_columns"]), (48, ["sales"]))
        ev = ts.evaluate_part(model, p.series, meta, "test", "cpu")
        self.assertAlmostEqual(ev["metrics"]["mae"], res["test_mae"], places=4)
        curve = ts.forecast_curve(p.series, ev, meta)
        self.assertEqual(len(curve["actual"]), len(curve["forecast"]))
        self.assertIsNone(curve["forecast"][0])


if __name__ == "__main__":
    unittest.main()
