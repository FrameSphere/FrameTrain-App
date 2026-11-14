"""Test-Plugin fuer Zeitreihen-Vorhersage (task_type 'time_series').

Bewertet den letzten Zeitabschnitt jeder Serie (gleiche zeitliche Aufteilung
wie im Training) und vergleicht mit der naiven Vorhersage. Die Kurve
"Vorhersage gegen echte Werte" geht als forecast_curve an die Oberflaeche.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "train_engine" / "plugins" / "time_series"))

from core.config import TestConfig  # noqa: E402
from core.protocol import TestProtocol  # noqa: E402

import ts_core as ts  # noqa: E402

SUPPORTED_ARCHITECTURES = ["patchtst", "ft_time_series"]


class Plugin:
    def __init__(self, config: TestConfig):
        self.config = config
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def setup(self) -> None:
        TestProtocol.status("init", "Lade Zeitreihenmodell...")
        self.model, self.meta = ts.load_trained(self.config.model_path, "cpu")

    def run_single(self) -> None:
        import numpy as np
        import torch
        t0 = time.time()
        L, H = int(self.meta["context_length"]), int(self.meta["prediction_length"])
        hist = ts.parse_history(self.config.single_input, len(self.meta["value_columns"]))
        note = ""
        if len(hist) < L:
            note = f"Nur {len(hist)} von {L} Kontextwerten — vorne mit dem ersten Wert aufgefuellt."
            hist = np.concatenate([np.repeat(hist[:1], L - len(hist), axis=0), hist])
        hist = hist[-L:]
        mu, sd = ts.Normalizer().stats("__eingabe__", hist)
        with torch.no_grad():
            z = self.model(past_values=torch.from_numpy(((hist - mu) / sd)[None].astype(np.float32))).prediction_outputs
        pred = z.numpy()[0] * sd + mu
        cols = self.meta["value_columns"]
        if len(cols) == 1:
            text = ", ".join(f"{v:.4g}" for v in pred[:, 0])
        else:
            text = "; ".join(", ".join(f"{v:.4g}" for v in row) for row in pred)
        TestProtocol.complete_single(
            predicted=text, confidence=None, top_predictions=[], inference_time=time.time() - t0,
            extra={"forecast": pred.tolist(), "history": hist.tolist(), "value_columns": cols, "note": note})

    def run_dataset(self) -> None:
        out_dir = Path(self.config.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        pc = self.config.plugin_config or {}
        part = str(pc.get("eval_part", "test"))
        df = ts.load_all(self.config.dataset_path)
        cols = ts.detect_columns(df, self.meta.get("time_column") or "", ",".join(self.meta["value_columns"]),
                                 self.meta.get("id_column") or "")
        series = ts.build_series(df, cols)
        need = int(self.meta["context_length"]) + int(self.meta["prediction_length"]) * 3
        series = {s: a for s, a in series.items() if len(a) >= need}
        if not series:
            raise ValueError(f"Keine Serie hat die noetigen {need} Zeitpunkte.")
        TestProtocol.status("testing", f"Bewerte den Abschnitt '{part}' von {len(series)} Serie(n)")
        t0 = time.time()
        ev = ts.evaluate_part(self.model, series, self.meta, part, "cpu",
                              int(self.config.max_samples) if self.config.max_samples else None)
        elapsed = time.time() - t0
        n = len(ev["meta"])
        if n == 0:
            raise ValueError("Der Testabschnitt ist kuerzer als die Vorhersagelaenge — keine Bewertung moeglich.")
        TestProtocol.progress(n, n, sps=n / max(elapsed, 1e-6), eta=0)
        metrics = dict(ev["metrics"])
        metrics.update({f"naive_{k}": v for k, v in ev["naive_metrics"].items()})
        # Anteil, um den das Modell den naiven Fehler senkt (>0 = besser als naiv).
        nm = ev["naive_metrics"]["mae"]
        metrics["mae_skill"] = 1.0 - metrics["mae"] / nm if nm > 0 else 0.0
        # "Treffer" = Fenster, in denen das Modell den naiven Vergleichswert schlaegt.
        import numpy as np
        mae_w = np.abs(ev["pred"] - ev["true"]).mean(axis=(1, 2))
        naive_w = np.abs(ev["naive"] - ev["true"]).mean(axis=(1, 2))
        better = mae_w < naive_w
        predictions = [{
            "sample_id": i, "input_text": f"{sid} @ {start}",
            "expected_output": ", ".join(f"{v:.4g}" for v in ev["true"][i][:, 0]),
            "predicted_output": ", ".join(f"{v:.4g}" for v in ev["pred"][i][:, 0]),
            "is_correct": bool(better[i]), "confidence": None,
            "mae": float(mae_w[i]), "naive_mae": float(naive_w[i]), "inference_time": elapsed / n}
            for i, (sid, start) in enumerate(ev["meta"])]
        curve = ts.forecast_curve(series, ev, self.meta)
        results_file = out_dir / "results.json"
        results_file.write_text(json.dumps({"task": "time_series", "part": part, "metrics": metrics,
                                            "forecast_curve": curve, "predictions": predictions},
                                           ensure_ascii=False, indent=2), encoding="utf-8")
        TestProtocol.complete_dataset(
            results_file=str(results_file), total_samples=n, accuracy=None, correct=None, average_loss=None,
            average_inference_time=elapsed / n, samples_per_second=n / max(elapsed, 1e-6),
            metrics=metrics, extra={"forecast_curve": curve, "windows_better_than_naive": int(better.sum())})
