"""Robustness check for text classifiers: does the prediction survive small changes?

Runs the model on the test data as it is, then on changed copies of the same
texts — lower case, no punctuation, typos (swapped letters), extra spaces —
and reports accuracy for each variant and how often the prediction flips.
A model that drops sharply on typos will struggle with real user input.

Model: any Hugging Face sequence classification model or FrameTrain version.
Data: table (parquet/csv/tsv/jsonl) with a text and a label column; the test
split is used if there is one.
"""
import json, os, random, re, sys
from pathlib import Path

MAX_SAMPLES = 300
BATCH_SIZE = 32
MAX_LENGTH = 256
TYPO_RATE = 0.08         # share of words that get two letters swapped

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
    def split_of(f):
        parts = [p.lower() for p in f.relative_to(DATASET_PATH).parts]
        for name in ("test", "val", "valid", "validation", "dev", "train"):
            if name in parts[:-1] or f.stem.lower().startswith(name):
                return "test" if name == "test" else "val" if name != "train" else "train"
        return "all"
    by_split = {}
    for f in files:
        by_split.setdefault(split_of(f), []).append(f)
    chosen = by_split.get("test") or by_split.get("val") or by_split.get("all") or by_split["train"]
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
    return df[text_col].astype(str).tolist(), df[label_col].tolist(), chosen


def typos(text, rng):
    out = []
    for word in text.split(" "):
        if len(word) > 3 and rng.random() < TYPO_RATE:
            i = rng.randrange(1, len(word) - 2)
            word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        out.append(word)
    return " ".join(out)


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    texts, labels, files = read_rows()
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
    gold = [to_id(v) for v in labels]
    if all(g < 0 for g in gold):
        sys.exit(f"Labels do not match the model classes {list(id2label.values())}.")

    def predict(batch_texts):
        preds = []
        with torch.no_grad():
            for i in range(0, len(batch_texts), BATCH_SIZE):
                enc = tokenizer(batch_texts[i:i + BATCH_SIZE], return_tensors="pt", padding=True,
                                truncation=True, max_length=MAX_LENGTH)
                preds += model(**enc).logits.argmax(dim=-1).tolist()
        return preds

    rng = random.Random(42)
    variants = {
        "original": texts,
        "lower case": [t.lower() for t in texts],
        "no punctuation": [re.sub(r"[^\w\s]", "", t) for t in texts],
        "typos": [typos(t, rng) for t in texts],
        "extra spaces": [re.sub(r" ", "  ", t) for t in texts],
    }
    print(f"Model: {mdir.name} | classes: {', '.join(id2label.values())}")
    print(f"Data: {len(texts)} rows from {', '.join(f.name for f in files)}")
    print()
    print(f"{'variant':<18}{'accuracy':>10}{'change':>10}{'flipped':>10}")
    base_preds, base_acc, report = None, None, {}
    for name, vt in variants.items():
        preds = predict(vt)
        acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
        if base_preds is None:
            base_preds, base_acc = preds, acc
        flipped = sum(p != b for p, b in zip(preds, base_preds)) / len(preds)
        report[name] = {"accuracy": round(acc, 4), "flipped": round(flipped, 4)}
        print(f"{name:<18}{acc:>10.3f}{acc - base_acc:>+10.3f}{flipped:>10.1%}")
    worst = min((n for n in report if n != "original"), key=lambda n: report[n]["accuracy"])
    print()
    print(f"Weakest variant: {worst} ({report[worst]['accuracy']:.3f} vs {base_acc:.3f} on the original text).")
    example = next(((o, v) for o, v, p, b in zip(texts, variants[worst], predict(variants[worst]), base_preds) if p != b), None)
    if example:
        print(f"Example of a flipped prediction:\n  original: {example[0][:160]}\n  changed:  {example[1][:160]}")
    print(json.dumps({"type": "complete", "data": report}))


if __name__ == "__main__":
    main()
