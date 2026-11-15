"""Echter Mini-Lauf durch Train- und Test-Engine (Subprozess, wie die App sie startet).

Je Plugin: selbst erzeugte Daten -> train_engine.py --config -> test_engine.py
--config (Dataset und Einzeltest). Geprueft wird das JSON-Protokoll auf stdout
und dass das Ergebnis den Vergleichswert schlaegt.

Aufruf:  python3.11 test_tabular_timeseries_e2e.py -v   (aus train_engine/)
Nur CPU, winzige Modelle; dauert etwa eine halbe Minute.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train_engine.py"
TEST = HERE.parent / "test_engine" / "test_engine.py"
sys.path.insert(0, str(HERE / "plugins" / "time_series"))
sys.path.insert(0, str(HERE / "plugins" / "tabular"))

from plugins.time_series.test_time_series import make_series_frame  # noqa: E402
from plugins.tabular.test_tabular import make_frame  # noqa: E402


def run_engine(script: Path, config: dict, workdir: Path, name: str):
    cfg_path = workdir / f"{name}.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script), "--config", str(cfg_path)],
                          capture_output=True, text=True, timeout=600)
    msgs = []
    for line in proc.stdout.splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(m, dict) and "type" in m:
            msgs.append(m)
    errors = [m for m in msgs if m["type"] == "error"]
    if errors:
        raise AssertionError(f"{name}: {errors[0]['data']}\nstderr:\n{proc.stderr[-2000:]}")
    complete = [m for m in msgs if m["type"] == "complete"]
    if not complete:
        raise AssertionError(f"{name}: kein complete\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}")
    return complete[-1]["data"], msgs


def blank_model(root: Path, model_type: str) -> Path:
    """Wie der Knopf 'Leeres Modell anlegen': Ordner mit config.json, ohne Gewichte."""
    d = root / f"blank_{model_type}"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return d


def base_train_cfg(model: Path, dataset: Path, out: Path, task: str, plugin_config: dict, **kw) -> dict:
    cfg = {"model_path": str(model), "dataset_path": str(dataset), "output_path": str(out / "final_model"),
           "checkpoint_dir": str(out / "checkpoints"), "task_type": task, "plugin_config": plugin_config,
           "epochs": 10, "batch_size": 32, "learning_rate": 3e-3, "weight_decay": 0.0, "seed": 7}
    cfg.update(kw)
    return cfg


class TabularEngineRun(unittest.TestCase):
    def test_train_und_test_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "kunden"
            for name, df in (("train", make_frame(600, seed=3)), ("test", make_frame(200, seed=4))):
                (data / name).mkdir(parents=True)
                df.to_csv(data / name / f"{name}.csv", index=False)
            model = blank_model(root, "ft_tabular")
            done, msgs = run_engine(TRAIN, base_train_cfg(model, data, root / "job", "tabular",
                                                          {"target_column": "label", "patience": 0}, epochs=30),
                                    root, "train_tab")
            self.assertTrue(any(m["type"] == "progress" for m in msgs))
            m = done["final_metrics"]
            print(f"\n[tabular train] test_accuracy={m['test_accuracy']:.3f} "
                  f"baseline={m['test_baseline_accuracy']:.3f} macro_f1={m['test_macro_f1']:.3f}")
            version = Path(done["model_path"])
            test_cfg = {"model_path": str(version), "dataset_path": str(data), "output_path": str(root / "test1"),
                        "task_type": "tabular", "mode": "dataset", "plugin_config": {}}
            res, _ = run_engine(TEST, test_cfg, root, "test_tab")
            print(f"[tabular test ] accuracy={res['accuracy']:.3f} metrics={res['metrics']}")
            self.assertGreater(res["accuracy"], res["metrics"]["baseline_accuracy"] + 0.15)
            self.assertEqual(res["total_samples"], 200)
            single, _ = run_engine(TEST, {**test_cfg, "mode": "single", "output_path": str(root / "t2"),
                                          "single_input": "x1=2.5; x2=40; city=Mainz"}, root, "single_tab")
            print(f"[tabular single] {single['predicted_output']} ({single['confidence']:.2f})")
            self.assertEqual(single["predicted_output"], "ja")

    def test_regression_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_frame(700, seed=5).drop(columns=["label"]).to_parquet(root / "preise.parquet")
            model = blank_model(root, "ft_tabular")
            done, _ = run_engine(TRAIN, base_train_cfg(model, root / "preise.parquet", root / "job", "tabular",
                                                       {"target_column": "price", "patience": 0}, epochs=40),
                                 root, "train_reg")
            test_cfg = {"model_path": done["model_path"], "dataset_path": str(root / "preise.parquet"),
                        "output_path": str(root / "test"), "task_type": "tabular", "mode": "dataset"}
            res, _ = run_engine(TEST, test_cfg, root, "test_reg")
            m = res["metrics"]
            print(f"\n[tabular regression test] mae={m['mae']:.3f} baseline_mae={m['baseline_mae']:.3f} r2={m['r2']:.3f}")
            self.assertLess(m["mae"], 0.5 * m["baseline_mae"])
            single, _ = run_engine(TEST, {**test_cfg, "mode": "single", "single_input": '{"x1": 1, "x2": 50, "city": "Berlin"}'},
                                   root, "single_reg")
            # Wahrer Erwartungswert: 3*1 + 0.05*50 + 4 = 9.5
            print(f"[tabular regression single] {single['predicted_output']} (erwartet ~9.5)")
            self.assertLess(abs(float(single["predicted_output"]) - 9.5), 1.5)


class TimeSeriesEngineRun(unittest.TestCase):
    def check(self, model: Path, root: Path, plugin_config: dict, label: str):
        data = root / f"data_{label}"
        data.mkdir()
        make_series_frame(500, ids=("a", "b"), missing=0.02, seed=11).to_csv(data / "sales.csv", index=False)
        done, msgs = run_engine(TRAIN, base_train_cfg(model, data, root / f"job_{label}", "time_series",
                                                      {"device": "cpu", "patience": 0, **plugin_config}, epochs=8),
                                root, f"train_{label}")
        self.assertTrue(any(m["type"] == "progress" for m in msgs))
        version = Path(done["model_path"])
        test_cfg = {"model_path": str(version), "dataset_path": str(data), "output_path": str(root / f"test_{label}"),
                    "task_type": "time_series", "mode": "dataset"}
        res, _ = run_engine(TEST, test_cfg, root, f"test_{label}")
        m = res["metrics"]
        print(f"\n[time_series {label}] windows={res['total_samples']} mae={m['mae']:.3f} naive_mae={m['naive_mae']:.3f} "
              f"rmse={m['rmse']:.3f} smape={m['smape']:.2f}% mase={m['mase']:.3f} naive_mase={m['naive_mase']:.3f}")
        self.assertLess(m["mae"], m["naive_mae"])
        self.assertLess(m["mase"], m["naive_mase"])
        curve = res["forecast_curve"]
        self.assertEqual(len(curve["actual"]), len(curve["forecast"]))
        values = ", ".join(str(round(v, 2)) for v in make_series_frame(500, ids=("a",), seed=11)["sales"][-80:])
        single, _ = run_engine(TEST, {**test_cfg, "mode": "single", "single_input": values}, root, f"single_{label}")
        n = len(single["predicted_output"].split(","))
        print(f"[time_series {label} single] {n} Werte: {single['predicted_output'][:80]}...")
        self.assertEqual(n, json.loads((version / "frametrain_timeseries.json").read_text())["prediction_length"])

    def test_neu_initialisiert(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.check(blank_model(root, "ft_time_series"), root,
                       {"context_length": 48, "prediction_length": 12, "d_model": 32, "ffn_dim": 64}, "scratch")

    def test_von_vortrainiertem_patchtst(self):
        """Simuliert einen HF-Download: PatchTST mit Gewichten, anderer Vorhersagelaenge."""
        from transformers import PatchTSTConfig, PatchTSTForPrediction
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf = root / "hf_patchtst"
            PatchTSTForPrediction(PatchTSTConfig(num_input_channels=7, context_length=64, prediction_length=24,
                                                 patch_length=8, patch_stride=8, d_model=32, num_hidden_layers=2,
                                                 num_attention_heads=4, ffn_dim=64)).save_pretrained(str(hf))
            # context_length 48 wird ignoriert: die Kontextlaenge des Basismodells (64) gilt.
            self.check(hf, root, {"context_length": 48, "prediction_length": 12}, "pretrained")
            meta = next(root.glob("job_pretrained/final_model/frametrain_timeseries.json"))
            data = json.loads(meta.read_text())
            self.assertEqual((data["context_length"], data["source"]), (64, "huggingface"))


if __name__ == "__main__":
    unittest.main()
