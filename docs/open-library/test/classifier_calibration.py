"""Calibration check: can you trust the confidence of your text classifier?

A model that says "95 % sure" should be right about 95 % of the time. This
script compares confidence with actual accuracy in ten bins, reports the
Expected Calibration Error (ECE) and finds a temperature that fixes most of the
gap (divide the logits by it before the softmax).

Model: any Hugging Face sequence classification model or FrameTrain version.
Data: table (parquet/csv/tsv/jsonl) with a text and a label column; the test
split is used if there is one.
"""
import json, os, sys
from pathlib import Path

MAX_SAMPLES = 500
BATCH_SIZE = 32
MAX_LENGTH = 256
BINS = 10

MODEL_PATH = Path(os.environ["MODEL_PATH"])
DATASET_PATH = Path(os.environ["DATASET_PATH"])


def model_dir():
    if (MODEL_PATH / "config.json").exists():
        return MODEL_PATH
    hits = sorted(MODEL_PATH.rglob("config.json"))
    if not hits:
        sys.exit(f"No config.json found in {MODEL_PATH}")
    return hits[0].parent


def read_rows():
    import pandas as pd
    files = [f for f in sorted(DATASET_PATH.rglob("*")) if f.is_file()
             and f.suffix.lower() in (".parquet", ".csv", ".tsv", ".jsonl")
             and not any(p.startswith(".") for p in f.relative_to(DATASET_PATH).parts)]
    if not files:
        sys.exit(f"No table files found in {DATASET_PATH}")
    def is_split(f, names):
        parts = [p.lower() for p in f.relative_to(DATASET_PATH).parts]
        return any(n in parts[:-1] or f.stem.lower().startswith(n) for n in names)
    chosen = ([f for f in files if is_split(f, ("test",))]
              or [f for f in files if is_split(f, ("val", "valid", "validation", "dev"))]
              or [f for f in files if not is_split(f, ("train",))] or files)
    frames = []
    for f in chosen:
        ext = f.suffix.lower()
        frames.append(pd.read_parquet(f) if ext == ".parquet" else
                      pd.read_json(f, lines=True) if ext == ".jsonl" else
                      pd.read_csv(f, sep="\t" if ext == ".tsv" else ","))
    df = pd.concat(frames, ignore_index=True)
    text_col = next((c for c in ("text", "sentence", "content", "review") if c in df.columns), None)
    label_col = next((c for c in ("label", "labels", "class", "category", "sentiment") if c in df.columns), None)
    if text_col is None or label_col is None:
        sys.exit(f"Need a text and a label column, found: {list(df.columns)}")
    df = df[[text_col, label_col]].dropna()
    if len(df) > MAX_SAMPLES:
        df = df.sample(MAX_SAMPLES, random_state=42)
    return df[text_col].astype(str).tolist(), df[label_col].tolist()


def ece(conf, correct, bins=BINS):
    import numpy as np
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            acc, avg = float(correct[mask].mean()), float(conf[mask].mean())
            total += mask.mean() * abs(acc - avg)
            rows.append((lo, hi, int(mask.sum()), avg, acc))
    return float(total), rows


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    texts, labels = read_rows()
    mdir = model_dir()
    tokenizer = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).eval()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    name_to_id = {v.lower(): k for k, v in id2label.items()}

    def to_id(v):
        if isinstance(v, str) and v.lower() in name_to_id:
            return name_to_id[v.lower()]
        try:
            return int(v)
        except (TypeError, ValueError):
            return -1
    gold = np.array([to_id(v) for v in labels])
    if (gold < 0).all():
        sys.exit(f"Labels do not match the model classes {list(id2label.values())}.")

    logits = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            enc = tokenizer(texts[i:i + BATCH_SIZE], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            logits.append(model(**enc).logits)
    logits = torch.cat(logits)

    def measure(temperature):
        probs = torch.softmax(logits / temperature, dim=-1).numpy()
        conf, pred = probs.max(axis=1), probs.argmax(axis=1)
        return conf, (pred == gold).astype(float)

    conf, correct = measure(1.0)
    value, rows = ece(conf, correct)
    print(f"Model: {mdir.name} | {len(texts)} rows | accuracy {correct.mean():.3f} | mean confidence {conf.mean():.3f}")
    print()
    print(f"{'confidence':<14}{'rows':>6}{'avg conf':>10}{'accuracy':>10}{'gap':>8}")
    for lo, hi, n, avg, acc in rows:
        print(f"{lo:.1f} - {hi:.1f}{'':<5}{n:>6}{avg:>10.3f}{acc:>10.3f}{acc - avg:>+8.3f}")
    print()
    verdict = ("overconfident (confidence above accuracy)" if conf.mean() > correct.mean() + 0.02 else
               "underconfident (confidence below accuracy)" if conf.mean() < correct.mean() - 0.02 else "well calibrated on average")
    print(f"ECE: {value:.4f}  ->  {verdict}")

    # Temperature scaling: one number, found by a simple search on these rows.
    best_t, best_ece = 1.0, value
    for t in np.arange(0.5, 5.01, 0.05):
        e, _ = ece(*measure(float(t)))
        if e < best_ece:
            best_t, best_ece = float(t), e
    if abs(best_t - 1.0) < 1e-6:
        print("No temperature between 0.5 and 5 lowers the ECE: nothing to correct.")
    else:
        print(f"Temperature {best_t:.2f} lowers the ECE to {best_ece:.4f} (accuracy stays the same).")
        print("Tuned on the evaluated rows; confirm it on separate data before relying on it.")
    print(json.dumps({"type": "complete", "data": {"ece": round(value, 4), "accuracy": round(float(correct.mean()), 4),
                                                   "mean_confidence": round(float(conf.mean()), 4),
                                                   "temperature": round(best_t, 2), "ece_after_scaling": round(best_ece, 4)}}))


if __name__ == "__main__":
    main()
