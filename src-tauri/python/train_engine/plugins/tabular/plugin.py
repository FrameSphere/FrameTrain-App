"""
plugins/tabular/plugin.py
=========================
Tabular-Daten-Plugin für FrameTrain.

Unterstützt:
  - Klassifikation: XGBoost, LightGBM, RandomForest, LogisticRegression, SVM
  - Regression:     XGBoost, LightGBM, RandomForestRegressor, LinearRegression

Dataset-Format:
  - CSV / TSV / Parquet / JSON mit Header
  - Zielspalte: "label", "target", "class", "y" oder letzte Spalte

Modell-Auswahl über model_path (Algorithmus-Name):
  "xgboost" / "xgb"          → XGBoost
  "lightgbm" / "lgbm"        → LightGBM
  "random_forest" / "rf"     → Scikit-learn RandomForest
  "logistic_regression" / "lr" → Scikit-learn LogisticRegression
  "linear_regression"        → Scikit-learn LinearRegression
  "svm"                      → Scikit-learn SVM

metrics.json Format: flat auf Root-Level (Rust-kompatibel).
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import TrainingConfig
from core.metrics import MetricsCollector
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol


def _detect_task(y) -> str:
    import numpy as np
    unique = np.unique(y)
    if len(unique) <= 20 or y.dtype == object or str(y.dtype).startswith("int"):
        return "classification"
    return "regression"


def _resolve_algorithm(model_path: str) -> str:
    p = model_path.lower().replace("-", "_").replace(" ", "_")
    if any(k in p for k in ["xgboost", "xgb"]):            return "xgboost"
    if any(k in p for k in ["lightgbm", "lgbm"]):          return "lightgbm"
    if any(k in p for k in ["random_forest", "rf"]):       return "random_forest"
    if any(k in p for k in ["logistic_regression", "logreg", "logistic"]): return "logistic_regression"
    if any(k in p for k in ["linear_regression", "linear"]): return "linear_regression"
    if any(k in p for k in ["svm", "svc", "svr"]):         return "svm"
    return "xgboost"


class Plugin(TrainPlugin):
    """Tabular-Plugin — XGBoost / LightGBM / sklearn."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._start_time = time.time()
        self.algorithm: str = "xgboost"
        self.task: str = "classification"
        self.model = None
        self.X_train = None; self.y_train = None
        self.X_val   = None; self.y_val   = None
        self.feature_names: List[str] = []
        self.label_col: str = "label"
        self.metrics = MetricsCollector()

    def setup(self) -> None:
        import random
        import numpy as np
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        self.algorithm = _resolve_algorithm(self.config.model_path)
        MessageProtocol.status("init", f"Algorithmus: {self.algorithm}")
        MessageProtocol.status("device", "CPU (Tabular-Training)")

    def load_data(self) -> None:
        import pandas as pd
        import numpy as np

        cfg  = self.config
        root = Path(cfg.dataset_path)

        train_file = self._find_data_file(root / "train") or self._find_data_file(root)
        if not train_file:
            raise FileNotFoundError(
                f"Keine Tabular-Datei in: {root}\n"
                "Unterstützt: .csv, .tsv, .parquet, .json, .jsonl"
            )

        MessageProtocol.status("loading", f"Lade: {train_file.name}")
        train_df = self._read_file(train_file)
        MessageProtocol.status("loading", f"{len(train_df):,} Zeilen | {len(train_df.columns)} Spalten")

        self.label_col = self._find_label_col(train_df)
        MessageProtocol.status("loading", f"Zielspalte: '{self.label_col}'")

        X = pd.get_dummies(train_df.drop(columns=[self.label_col]))
        y = train_df[self.label_col]
        self.feature_names = list(X.columns)
        self.X_train = X.values.astype(float)
        self.y_train = y.values

        task_hint = cfg.task_type.lower()
        if task_hint == "regression":        self.task = "regression"
        elif task_hint == "classification":  self.task = "classification"
        else:                                self.task = _detect_task(self.y_train)
        MessageProtocol.status("loading", f"Aufgabe: {self.task}")

        val_file = (self._find_data_file(root / "val")
                    or self._find_data_file(root / "validation"))
        if val_file:
            val_df = self._read_file(val_file)
            Xv = pd.get_dummies(val_df.drop(columns=[self.label_col], errors="ignore"))
            Xv = Xv.reindex(columns=self.feature_names, fill_value=0)
            self.X_val = Xv.values.astype(float)
            self.y_val = val_df.get(self.label_col, pd.Series()).values
            MessageProtocol.status("loading", f"Validierung: {len(val_df):,} Zeilen")
        else:
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.1, random_state=cfg.seed
            )
            MessageProtocol.status(
                "loading",
                f"Auto-Split: {len(self.X_train):,} Train / {len(self.X_val):,} Val"
            )

    def _find_data_file(self, path: Path) -> Optional[Path]:
        if not path.exists():
            return None
        if path.is_file():
            return path
        for ext in [".parquet", ".csv", ".tsv", ".json", ".jsonl"]:
            found = sorted(path.glob(f"*{ext}"))
            if found:
                return found[0]
        return None

    def _read_file(self, path: Path):
        import pandas as pd
        ext = path.suffix.lower()
        if ext == ".parquet":  return pd.read_parquet(path)
        if ext == ".csv":      return pd.read_csv(path)
        if ext == ".tsv":      return pd.read_csv(path, sep="\t")
        if ext in (".json", ".jsonl"): return pd.read_json(path, lines=(ext == ".jsonl"))
        raise ValueError(f"Unbekanntes Format: {ext}")

    def _find_label_col(self, df) -> str:
        for c in ["label", "labels", "target", "class", "y", "output"]:
            if c in df.columns:
                return c
        return df.columns[-1]

    def build_model(self) -> None:
        alg = self.algorithm
        cfg = self.config
        n   = max(100, cfg.epochs * 10)

        if alg == "xgboost":
            import xgboost as xgb
            p = dict(n_estimators=n, learning_rate=cfg.learning_rate, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, seed=cfg.seed, verbosity=0, n_jobs=-1)
            self.model = (xgb.XGBClassifier(**p, use_label_encoder=False, eval_metric="logloss")
                          if self.task == "classification" else xgb.XGBRegressor(**p))

        elif alg == "lightgbm":
            import lightgbm as lgb
            p = dict(n_estimators=n, learning_rate=cfg.learning_rate, num_leaves=31,
                     seed=cfg.seed, verbosity=-1, n_jobs=-1)
            self.model = (lgb.LGBMClassifier(**p) if self.task == "classification"
                          else lgb.LGBMRegressor(**p))

        elif alg == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            p = dict(n_estimators=n, random_state=cfg.seed, n_jobs=-1)
            self.model = (RandomForestClassifier(**p) if self.task == "classification"
                          else RandomForestRegressor(**p))

        elif alg == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(
                C=1.0 / max(cfg.weight_decay, 1e-9), max_iter=cfg.epochs * 100,
                random_state=cfg.seed, n_jobs=-1)

        elif alg == "linear_regression":
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression(n_jobs=-1)

        elif alg == "svm":
            from sklearn.svm import SVC, SVR
            self.model = (SVC if self.task == "classification" else SVR)(
                C=1.0 / max(cfg.weight_decay, 1e-9), kernel="rbf")
        else:
            raise ValueError(f"Unbekannter Algorithmus: {alg}")

        MessageProtocol.status(
            "loading",
            f"{alg} ({self.task}) | {len(self.X_train):,} Samples | {len(self.feature_names)} Features"
        )

    def train(self) -> None:
        cfg = self.config
        MessageProtocol.status("training", f"Starte {self.algorithm} Training...")
        MessageProtocol.progress(epoch=1, total_epochs=1, step=0, total_steps=1, train_loss=0.0)

        fit_kwargs: dict = {}
        if self.algorithm in ("xgboost", "lightgbm") and self.X_val is not None:
            fit_kwargs["eval_set"] = [(self.X_val, self.y_val)]
            if self.algorithm == "xgboost":
                fit_kwargs["verbose"] = False

        self.model.fit(self.X_train, self.y_train, **fit_kwargs)

        elapsed = round(time.time() - self._start_time, 1)
        MessageProtocol.status("training", f"Training in {elapsed}s abgeschlossen")
        MessageProtocol.progress(epoch=1, total_epochs=1, step=1, total_steps=1, train_loss=0.0)
        self.metrics.record(1, 1, {"training_seconds": elapsed})
        self.metrics.total_epochs = 1
        self.metrics.total_steps  = 1

    def validate(self) -> Dict[str, float]:
        duration = int(time.time() - self._start_time)
        extra: Dict[str, Any] = {
            "total_epochs":              1,
            "total_steps":               1,
            "training_duration_seconds": duration,
        }

        if self.model is None or self.X_val is None:
            return self.metrics.final_metrics(extra)

        y_pred = self.model.predict(self.X_val)

        if self.task == "classification":
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(self.y_val, y_pred)
            f1  = f1_score(self.y_val, y_pred, average="weighted", zero_division=0)
            extra.update({"val_accuracy": round(acc, 4), "val_f1": round(f1, 4)})
            MessageProtocol.status("validating", f"Accuracy: {acc:.4f}  F1: {f1:.4f}")
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(self.y_val, y_pred)
            r2  = r2_score(self.y_val, y_pred)
            extra.update({"val_mse": round(mse, 6), "val_r2": round(r2, 4)})
            MessageProtocol.status("validating", f"MSE: {mse:.6f}  R²: {r2:.4f}")

        return self.metrics.final_metrics(extra)

    def export(self) -> str:
        import joblib
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, out / "model.pkl")

        meta = {
            "algorithm":     self.algorithm,
            "task":          self.task,
            "feature_names": self.feature_names,
            "label_col":     self.label_col,
            "n_features":    len(self.feature_names),
        }
        (out / "config.json").write_text(json.dumps(meta, indent=2))

        # metrics.json flat (Rust-kompatibel)
        final = self.validate()
        self.metrics.save_with_overrides(str(out), final)

        MessageProtocol.status("saved", f"Modell gespeichert: {out}")
        return str(out)
