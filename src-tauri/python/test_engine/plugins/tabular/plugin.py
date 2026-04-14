"""
plugins/tabular/plugin.py
=========================
Test-Plugin für Tabular-Modelle (sklearn, XGBoost, LightGBM, CatBoost).

Dataset-Format:
  dataset_path/test/data.csv   – Spalte 'label' oder letzte Spalte = Ziel
  dataset_path/test/X_test.csv + y_test.csv (getrennte Dateien)

Single-Modus:
  input_data: JSON-String z. B. '{"feature1": 1.0, "feature2": "A"}'
  Gibt Vorhersage + ggf. Klassen-Wahrscheinlichkeiten zurück.

Metriken Klassifikation: Accuracy, F1 (macro), Precision, Recall
Metriken Regression:     R², MAE, RMSE
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import TestConfig
from core.plugin_base import TestPlugin
from core.protocol import MessageProtocol


class Plugin(TestPlugin):

    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.is_regression: bool = False
        self.feature_names: List[str] = []
        self.label_encoder = None

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        self.device = "cpu"
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        self.proto.status("init", "Tabular-Plugin | device=cpu")

    # ── Modell laden ───────────────────────────────────────────────────────

    def load_model(self) -> None:
        import pickle
        model_path = Path(self.config.model_path)
        self.proto.status("loading", f"Lade Tabular-Modell: {model_path.name} …")

        # Modell-Datei finden
        pkl_files = list(model_path.glob("*.pkl")) + list(model_path.glob("*.joblib"))
        if not pkl_files:
            raise FileNotFoundError(
                f"Kein Tabular-Modell (.pkl/.joblib) gefunden in: {model_path}\n"
                f"Das train_engine Tabular-Plugin speichert Modelle als model.pkl"
            )

        model_file = next((f for f in pkl_files if "model" in f.stem), pkl_files[0])

        import joblib
        self.model = joblib.load(str(model_file))

        # Modell-Info laden
        info_file = model_path / "model_info.json"
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text())
                task = info.get("task_type", self.config.task_type).lower()
                self.is_regression = "regression" in task
                self.feature_names = info.get("feature_names", [])
            except Exception:
                pass

        # Fallback aus config
        if "regression" in self.config.task_type.lower():
            self.is_regression = True

        # Optional: LabelEncoder laden
        enc_file = next((f for f in pkl_files if "encoder" in f.stem or "label" in f.stem), None)
        if enc_file:
            self.label_encoder = joblib.load(str(enc_file))

        task_desc = "Regression" if self.is_regression else "Klassifikation"
        self.proto.status("loaded", f"Tabular-Modell geladen ({task_desc})")

    # ── DataFrame-Vorverarbeitung ──────────────────────────────────────────

    def _to_dataframe(self, rows: List[Dict]) -> Any:
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            if self.feature_names:
                cols = [c for c in self.feature_names if c in df.columns]
                if cols:
                    df = df[cols]
            return df
        except ImportError:
            raise ImportError("pandas nicht installiert:\n  pip install pandas")

    # ── Dataset laden ──────────────────────────────────────────────────────

    def _load_test_data(self):
        import pandas as pd
        root = Path(self.config.dataset_path)
        test_dir = root / "test"
        if not test_dir.exists():
            test_dir = root / "val"
        if not test_dir.exists():
            test_dir = root

        # Variante A: X_test.csv + y_test.csv
        x_file = test_dir / "X_test.csv"
        y_file = test_dir / "y_test.csv"
        if x_file.exists() and y_file.exists():
            X = pd.read_csv(x_file)
            y = pd.read_csv(y_file).squeeze()
            return X, y

        # Variante B: data.csv / test.csv mit label-Spalte
        for fname in ["data.csv", "test.csv", "test_data.csv"]:
            csv_f = test_dir / fname
            if csv_f.exists():
                df = pd.read_csv(csv_f)
                target_col = None
                for candidate in ["label", "target", "y", "class", "output"]:
                    if candidate in df.columns:
                        target_col = candidate
                        break
                if target_col is None:
                    target_col = df.columns[-1]
                y = df[target_col]
                X = df.drop(columns=[target_col])
                return X, y

        # Variante C: Einzige CSV im Ordner
        csvs = list(test_dir.glob("*.csv"))
        if csvs:
            df = pd.read_csv(csvs[0])
            target_col = df.columns[-1]
            return df.drop(columns=[target_col]), df[target_col]

        raise ValueError(f"Keine Test-Daten gefunden in: {test_dir}")

    # ── Dataset-Modus ─────────────────────────────────────────────────────

    def run_dataset(self) -> Dict[str, Any]:
        self.proto.status("loading", "Lade Test-Daten …")
        X, y = self._load_test_data()

        if self.config.max_samples:
            X = X.head(self.config.max_samples)
            y = y.head(self.config.max_samples)

        total = len(X)
        self.proto.status("testing", f"Teste {total} Samples …")

        t_start = time.time()
        results: List[Dict[str, Any]] = []

        # Batch-Inferenz
        batch_size = max(self.config.batch_size, 32)
        for batch_start in range(0, total, batch_size):
            if self.is_stopped:
                break
            batch_end = min(batch_start + batch_size, total)
            X_batch = X.iloc[batch_start:batch_end]
            y_batch = y.iloc[batch_start:batch_end].tolist()

            t0 = time.time()
            preds = self.model.predict(X_batch).tolist()
            proba = None
            if not self.is_regression and hasattr(self.model, "predict_proba"):
                try:
                    proba = self.model.predict_proba(X_batch).tolist()
                except Exception:
                    pass
            t_batch = (time.time() - t0) / len(preds)

            for i, (pred, true) in enumerate(zip(preds, y_batch)):
                sample_id = batch_start + i
                if self.is_regression:
                    is_correct = abs(float(pred) - float(true)) < 0.01
                else:
                    is_correct = str(pred).strip() == str(true).strip()

                r: Dict[str, Any] = {
                    "sample_id": sample_id,
                    "input_text": json.dumps(
                        {col: X_batch.iloc[i][col] for col in X_batch.columns[:10]},
                        default=str
                    ),
                    "predicted_output": str(pred),
                    "expected_output": str(true),
                    "is_correct": is_correct,
                    "loss": None,
                    "confidence": None,
                    "inference_time": t_batch,
                    "error_type": None,
                }
                if proba:
                    p_row = proba[i]
                    r["confidence"] = max(p_row)
                    r["probabilities"] = p_row
                results.append(r)

            elapsed = time.time() - t_start
            sps = (batch_end) / elapsed if elapsed > 0 else 0.0
            self.proto.progress(batch_end, total, sps)

        elapsed_total = time.time() - t_start

        # Metriken berechnen
        preds_all = [r["predicted_output"] for r in results]
        trues_all = [r["expected_output"] for r in results]

        metrics: Dict[str, Any] = {
            "total_samples": total,
            "average_inference_time": elapsed_total / max(total, 1),
            "samples_per_second": total / elapsed_total if elapsed_total > 0 else 0.0,
            "total_time": elapsed_total,
            "average_loss": None,
        }

        try:
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score, recall_score,
                mean_absolute_error, mean_squared_error, r2_score
            )
            import numpy as np

            if self.is_regression:
                t_f = [float(t) for t in trues_all]
                p_f = [float(p) for p in preds_all]
                metrics["r2_score"] = r2_score(t_f, p_f)
                metrics["mae"] = mean_absolute_error(t_f, p_f)
                metrics["rmse"] = float(np.sqrt(mean_squared_error(t_f, p_f)))
                metrics["accuracy"] = max(0.0, metrics["r2_score"] * 100)  # R² als Proxy
            else:
                correct = sum(1 for r in results if r["is_correct"])
                metrics["accuracy"] = accuracy_score(trues_all, preds_all) * 100
                metrics["correct_predictions"] = correct
                metrics["incorrect_predictions"] = total - correct
                try:
                    metrics["f1_macro"] = f1_score(trues_all, preds_all, average="macro", zero_division=0)
                    metrics["precision"] = precision_score(trues_all, preds_all, average="macro", zero_division=0)
                    metrics["recall"] = recall_score(trues_all, preds_all, average="macro", zero_division=0)
                except Exception:
                    pass
        except ImportError:
            correct = sum(1 for r in results if r["is_correct"])
            metrics["accuracy"] = (correct / total * 100) if total > 0 else 0.0
            metrics["correct_predictions"] = correct
            metrics["incorrect_predictions"] = total - correct

        out = Path(self.config.output_path)
        results_file = out / "test_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "predictions": results}, f, indent=2, default=str)

        hard = [r for r in results if not r["is_correct"]]
        hard_file = None
        if hard:
            hard_file = str(out / "hard_examples.jsonl")
            with open(hard_file, "w", encoding="utf-8") as f:
                for ex in hard:
                    f.write(json.dumps(ex, default=str) + "\n")

        return {
            "metrics": metrics,
            "predictions": results,
            "results_file": str(results_file),
            "hard_examples_file": hard_file,
            "total_samples": total,
            "task_type": "tabular",
        }

    # ── Single-Modus ───────────────────────────────────────────────────────

    def run_single(self, input_data: str) -> Dict[str, Any]:
        self.proto.status("inferring", "Tabular-Inferenz …")

        try:
            row = json.loads(input_data)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"single_input muss ein JSON-Objekt sein: {e}\n"
                f"Beispiel: {{\"feature1\": 1.0, \"feature2\": \"A\"}}"
            )

        import pandas as pd
        X = pd.DataFrame([row])
        if self.feature_names:
            missing = [c for c in self.feature_names if c not in X.columns]
            for m in missing:
                X[m] = 0
            X = X[[c for c in self.feature_names if c in X.columns]]

        t0 = time.time()
        pred = self.model.predict(X)[0]
        proba = None
        if not self.is_regression and hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X)[0].tolist()
            except Exception:
                pass
        inference_time = time.time() - t0

        result: Dict[str, Any] = {
            "prediction": str(pred),
            "inference_time": inference_time,
            "is_regression": self.is_regression,
        }
        if proba is not None:
            result["probabilities"] = proba
            result["confidence"] = max(proba)
        return result
