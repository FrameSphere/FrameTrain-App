"""Test-Plugin fuer Tabellenmodelle (task_type 'tabular').

Laedt Gewichte und Vorverarbeitung aus dem Versionsordner und nutzt denselben
Kern wie das Training (train_engine/plugins/tabular/tabular_core.py).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "train_engine" / "plugins" / "tabular"))

from core.config import TestConfig  # noqa: E402
from core.protocol import TestProtocol  # noqa: E402

import tabular_core as tc  # noqa: E402


def test_frame(dataset_path: str, split_info: dict):
    """Test-Datei, wenn vorhanden; sonst derselbe Test-Anteil wie beim Training."""
    frames = tc.load_frames(dataset_path)
    if "test" in frames:
        return frames["test"], "test"
    if "train" in frames:
        # Vorab geteilt, aber ohne test/: dann bleibt nur die Validierung ungesehen.
        if "val" in frames:
            return frames["val"], "val"
        return frames["train"], "train (keine Testdatei, Ergebnis ist geschoent)"
    split = split_info or {}
    splits = tc.resolve_splits(frames, float(split.get("val_ratio", 0.15)),
                               float(split.get("test_ratio", 0.15)), int(split.get("seed", 42)))
    return splits["test"], "test (gleiche Aufteilung wie im Training)"


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self) -> None:
        TestProtocol.status("init", "Lade Tabellenmodell...")
        self.model, self.prep = tc.load_bundle(self.config.model_path)
        meta_file = Path(self.config.model_path) / "model.json"
        self.meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}

    def run_single(self) -> None:
        t0 = time.time()
        row = tc.parse_row(self.config.single_input, self.prep.feature_columns)
        unknown = [k for k in row if k not in self.prep.feature_columns and k != self.prep.target]
        import pandas as pd
        out = tc.predict_frame(self.model, self.prep, pd.DataFrame([row]))
        missing = [c for c in self.prep.feature_columns if row.get(c) is None]
        extra = {"missing_columns": missing, "unknown_columns": unknown}
        if self.prep.task == "classification":
            proba = out["proba"][0]
            order = proba.argsort()[::-1]
            top = [{"label": self.prep.classes[i], "score": float(proba[i])} for i in order[:5]]
            TestProtocol.complete_single(predicted=self.prep.classes[int(order[0])], confidence=float(proba[order[0]]),
                                         top_predictions=top, inference_time=time.time() - t0, extra=extra)
        else:
            TestProtocol.complete_single(predicted=f"{float(out['pred'][0]):.6g}", confidence=None,
                                         top_predictions=[], inference_time=time.time() - t0, extra=extra)

    def run_dataset(self) -> None:
        import numpy as np
        out_dir = Path(self.config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        df, source = test_frame(self.config.dataset_path, self.meta.get("split", {}))
        if self.config.max_samples:
            df = df.iloc[: int(self.config.max_samples)].reset_index(drop=True)
        TestProtocol.status("testing", f"Teste {len(df)} Zeilen aus {source}")
        y, mask = self.prep.transform_target(df[self.prep.target]) if self.prep.target in df.columns \
            else (None, None)
        if y is None:
            raise ValueError(f"Zielspalte '{self.prep.target}' fehlt in den Testdaten.")
        df = df[mask].reset_index(drop=True)
        y = y[mask]
        t0 = time.time()
        out = tc.predict_frame(self.model, self.prep, df)
        elapsed = time.time() - t0
        total = len(df)
        TestProtocol.progress(total, total, sps=total / max(elapsed, 1e-6), eta=0)

        # Vergleichswert aus der Trainingsverteilung ist hier nicht greifbar;
        # haeufigste Klasse bzw. Mittelwert der Testdaten ist die ehrliche Untergrenze.
        if self.prep.task == "classification":
            metrics = tc.classification_metrics(y, out["pred"], len(self.prep.classes))
            metrics.update(tc.baseline_metrics(self.prep, y, y))
            correct = int(np.sum(out["pred"] == y))
            predictions = [{
                "sample_id": i, "input_text": json.dumps(_row(df, i), ensure_ascii=False, default=str)[:500],
                "expected_output": self.prep.classes[int(y[i])], "predicted_output": self.prep.classes[int(out["pred"][i])],
                "is_correct": bool(out["pred"][i] == y[i]), "confidence": float(out["proba"][i].max()),
                "inference_time": elapsed / max(total, 1)} for i in range(total)]
            accuracy = metrics["accuracy"]
        else:
            true = self.prep.decode_regression(y)
            metrics = tc.regression_metrics(true, out["pred"])
            metrics.update(tc.baseline_metrics(self.prep, true, true))
            correct = None
            accuracy = None
            predictions = [{
                "sample_id": i, "input_text": json.dumps(_row(df, i), ensure_ascii=False, default=str)[:500],
                "expected_output": f"{float(true[i]):.6g}", "predicted_output": f"{float(out['pred'][i]):.6g}",
                "is_correct": None, "confidence": None, "error": float(out["pred"][i] - true[i]),
                "inference_time": elapsed / max(total, 1)} for i in range(total)]
        metrics["total_samples"] = total
        results_file = out_dir / "results.json"
        results_file.write_text(json.dumps({"task": self.prep.task, "source": source, "metrics": metrics,
                                            "predictions": predictions}, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        TestProtocol.complete_dataset(
            results_file=str(results_file), total_samples=total, accuracy=accuracy, correct=correct,
            average_loss=None, average_inference_time=elapsed / max(total, 1),
            samples_per_second=total / max(elapsed, 1e-6),
            metrics={k: v for k, v in metrics.items() if k != "total_samples"},
            extra={"tabular_task": self.prep.task, "test_source": source})


def _row(df, i):
    return {k: (None if v != v else v) for k, v in df.iloc[i].to_dict().items()}  # NaN -> None
