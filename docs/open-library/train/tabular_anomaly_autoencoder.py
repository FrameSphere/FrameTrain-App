"""Anomaly detection on table data with an autoencoder (no labels needed).

Learns what a normal row looks like and scores every row by its reconstruction
error. Rows above the threshold are unusual: fraud candidates, sensor faults,
data entry errors. Works with a blank "Tabular" model or any model entry —
the base model's weights are not used.

Data: CSV / TSV / Parquet. Numbers are standardized, categories one-hot encoded,
ID-like and free-text columns are ignored. No label column required; a column
named in IGNORE_COLUMNS is left out (e.g. a known "is_fraud" flag you only want
for checking afterwards).

Saved to the version folder: autoencoder.pt, preprocessing.json (columns,
scaling, threshold) and anomaly_scores.csv with the score of every row.
"""
import json, os, sys
from pathlib import Path

EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
BOTTLENECK = 4               # size of the compressed representation
THRESHOLD_PERCENTILE = 97.5  # rows above this score percentile count as anomalies
IGNORE_COLUMNS = []          # e.g. ["is_fraud"]
MAX_CATEGORIES = 30          # text columns with more distinct values are ignored

DATASET_PATH = Path(os.environ["DATASET_PATH"])
OUTPUT_PATH = Path(os.environ["OUTPUT_PATH"])


def emit(kind, **data):
    print(json.dumps({"type": kind, "data": data}, ensure_ascii=False), flush=True)


def fail(message):
    emit("error", error=message)
    sys.exit(1)


def read_table():
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
        fail(f"No CSV, TSV or Parquet file found in {DATASET_PATH}")
    return pd.concat(frames, ignore_index=True)


def main():
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn

    df = read_table()
    emit("status", message=f"{len(df)} rows, {len(df.columns)} columns")
    numeric, categorical, skipped = [], [], []
    for col in df.columns:
        if col in IGNORE_COLUMNS:
            skipped.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # a number that is different in (almost) every row is an ID
            (skipped if df[col].nunique() >= 0.98 * len(df) and pd.api.types.is_integer_dtype(df[col]) else numeric).append(col)
        elif df[col].nunique() <= MAX_CATEGORIES:
            categorical.append(col)
        else:
            skipped.append(col)
    if not numeric and not categorical:
        fail("No usable columns: need numeric columns or text columns with few distinct values.")
    prep = {"numeric": {}, "categorical": {}, "skipped": skipped}
    parts = []
    for col in numeric:
        values = pd.to_numeric(df[col], errors="coerce")
        median = float(values.median())
        std = float(values.std()) or 1.0
        prep["numeric"][col] = {"median": median, "mean": float(values.mean()), "std": std}
        parts.append(((values.fillna(median) - float(values.mean())) / std).to_numpy(dtype="float32")[:, None])
    for col in categorical:
        cats = sorted(str(v) for v in df[col].dropna().unique())
        prep["categorical"][col] = cats
        text = df[col].astype(str)
        parts.append(np.stack([(text == c).to_numpy(dtype="float32") for c in cats], axis=1))
    x = torch.from_numpy(np.concatenate(parts, axis=1))
    dim = x.shape[1]
    emit("status", message=f"{len(numeric)} numeric and {len(categorical)} categorical columns -> {dim} inputs"
                           + (f"; ignored: {', '.join(skipped)}" if skipped else ""))

    hidden = max(dim // 2, BOTTLENECK * 2)
    model = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, BOTTLENECK), nn.ReLU(),
                          nn.Linear(BOTTLENECK, hidden), nn.ReLU(), nn.Linear(hidden, dim))
    torch.manual_seed(42)
    order = torch.randperm(len(x))
    cut = max(len(x) // 10, 1)
    val, train = x[order[:cut]], x[order[cut:]]
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    steps_per_epoch = max((len(train) + BATCH_SIZE - 1) // BATCH_SIZE, 1)
    step = 0
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(train))
        running = 0.0
        for i in range(0, len(train), BATCH_SIZE):
            batch = train[perm[i:i + BATCH_SIZE]]
            loss = nn.functional.mse_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running += loss.item()
            step += 1
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(model(val), val))
        emit("progress", epoch=epoch, total_epochs=EPOCHS, step=step, total_steps=EPOCHS * steps_per_epoch,
             train_loss=running / steps_per_epoch, val_loss=val_loss, learning_rate=LEARNING_RATE)

    with torch.no_grad():
        scores = ((model(x) - x) ** 2).mean(dim=1).numpy()
    threshold = float(np.percentile(scores, THRESHOLD_PERCENTILE))
    flagged = scores > threshold
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": dim, "hidden": hidden, "bottleneck": BOTTLENECK},
               OUTPUT_PATH / "autoencoder.pt")
    prep["threshold"] = threshold
    (OUTPUT_PATH / "preprocessing.json").write_text(json.dumps(prep, indent=2, ensure_ascii=False), encoding="utf-8")
    out = df.copy()
    out["anomaly_score"] = scores
    out["is_anomaly"] = flagged
    out.sort_values("anomaly_score", ascending=False).to_csv(OUTPUT_PATH / "anomaly_scores.csv", index=False)
    top = out.sort_values("anomaly_score", ascending=False).head(3).drop(columns=["is_anomaly"])
    emit("status", message=f"{int(flagged.sum())} of {len(df)} rows above the threshold {threshold:.4f}")
    emit("complete", val_loss=val_loss, threshold=threshold, anomalies=int(flagged.sum()), rows=len(df),
         inputs=dim, most_unusual_rows=json.loads(top.to_json(orient="records")))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
