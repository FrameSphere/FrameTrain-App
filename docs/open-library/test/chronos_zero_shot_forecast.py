"""Zero-shot forecasting with Chronos: a baseline without any training.

Chronos is a pretrained time series model. This script forecasts the last part
of your series with it and compares the error with a naive forecast (repeat the
last value / last season). Use it to see what a foundation model gives you for
free before you train your own model — and as the number your own model has
to beat.

Model: if the selected model is a Chronos checkpoint (amazon/chronos-bolt-*,
chronos-t5-*), it is used. Otherwise CHRONOS_MODEL is downloaded from
Hugging Face (chronos-bolt-tiny, about 35 MB).
Data: CSV / TSV / Parquet with one or more numeric columns, rows in time order
(a time column is used for sorting if there is one).
"""
import json, os, sys
from pathlib import Path

CHRONOS_MODEL = "amazon/chronos-bolt-tiny"
PREDICTION_LENGTH = 24
CONTEXT_LENGTH = 512
SEASON = 0               # season length for the naive forecast, 0 = repeat the last value
WINDOWS = 5              # how many forecasts are made, moving back through the series

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ""))
DATASET_PATH = Path(os.environ["DATASET_PATH"])


def read_series():
    import pandas as pd
    frames = []
    for f in sorted(DATASET_PATH.rglob("*")):
        ext = f.suffix.lower()
        if not f.is_file() or any(p.startswith(".") for p in f.relative_to(DATASET_PATH).parts):
            continue
        if ext == ".parquet":
            frames.append(pd.read_parquet(f))
        elif ext in (".csv", ".tsv"):
            frames.append(pd.read_csv(f, sep=None, engine="python"))
    if not frames:
        sys.exit(f"No CSV, TSV or Parquet file found in {DATASET_PATH}")
    df = pd.concat(frames, ignore_index=True)
    time_col = next((c for c in df.columns if str(c).lower() in ("timestamp", "time", "date", "datetime", "ds")), None)
    if time_col is not None:
        df = df.sort_values(time_col, key=lambda s: pd.to_datetime(s, errors="coerce"))
    cols = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        sys.exit(f"No numeric column found, columns: {list(df.columns)}")
    return {c: df[c].interpolate().bfill().to_numpy(dtype="float32") for c in cols}


def load_pipeline():
    from chronos import BaseChronosPipeline
    cfg = MODEL_PATH / "config.json"
    source = CHRONOS_MODEL
    if cfg.exists() and "chronos" in cfg.read_text(encoding="utf-8", errors="ignore").lower():
        source = str(MODEL_PATH)
    print(f"Chronos model: {source}")
    return BaseChronosPipeline.from_pretrained(source, device_map="cpu")


def main():
    import numpy as np
    import torch

    series = read_series()
    pipeline = load_pipeline()
    h = PREDICTION_LENGTH
    report = {}
    print()
    print(f"{'column':<20}{'windows':>8}{'MAE':>10}{'naive MAE':>11}{'MASE':>8}{'in 80% band':>13}")
    for name, values in series.items():
        if len(values) < 3 * h:
            print(f"{name:<20} skipped: only {len(values)} values, need at least {3 * h}")
            continue
        errors, naive_errors, covered = [], [], []
        for w in range(WINDOWS):
            end = len(values) - w * h
            if end - h < 2 * h:
                break
            context, actual = values[max(0, end - h - CONTEXT_LENGTH):end - h], values[end - h:end]
            quantiles, _ = pipeline.predict_quantiles(torch.tensor(context), prediction_length=h, quantile_levels=[0.1, 0.5, 0.9])
            low, median, high = (quantiles[0, :, i].numpy() for i in range(3))
            naive = np.resize(context[-SEASON:], h) if SEASON else np.full(h, context[-1])
            errors.append(np.abs(median - actual).mean())
            naive_errors.append(np.abs(naive - actual).mean())
            covered.append(((actual >= low) & (actual <= high)).mean())
        mae, naive_mae = float(np.mean(errors)), float(np.mean(naive_errors))
        mase = mae / naive_mae if naive_mae else float("nan")
        report[name] = {"mae": round(mae, 4), "naive_mae": round(naive_mae, 4), "mase": round(mase, 4),
                        "coverage_80": round(float(np.mean(covered)), 4), "windows": len(errors)}
        print(f"{str(name)[:19]:<20}{len(errors):>8}{mae:>10.4f}{naive_mae:>11.4f}{mase:>8.3f}{np.mean(covered):>13.1%}")
    if not report:
        sys.exit("No column was long enough to evaluate.")
    print()
    print("MASE below 1 means Chronos beats the naive forecast without any training.")
    print("'in 80% band' is how often the real value fell between the 10 % and 90 % quantile (ideal: about 80 %).")
    print(json.dumps({"type": "complete", "data": report}))


if __name__ == "__main__":
    main()
