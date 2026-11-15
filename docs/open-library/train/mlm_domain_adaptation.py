"""Domain adaptation with masked language modeling (BERT, RoBERTa, DistilBERT, ELECTRA ...).

Continue pre-training an encoder on your own unlabeled text, so a later
classification or NER fine-tuning starts from a model that knows your vocabulary.

Data: any table (parquet/csv/tsv/jsonl) with a text column, or plain .txt files
(one example per line). Labels are ignored. A val/ or test/ folder is used for
validation; otherwise 10 % of the rows are held out.

Settings: edit the constants below.
"""
import json, math, os, random, sys
from pathlib import Path

EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
MASK_PROBABILITY = 0.15
MAX_ROWS = 2000          # 0 = all rows
TEXT_COLUMN = ""         # empty = detect

MODEL_PATH = os.environ["MODEL_PATH"]
DATASET_PATH = Path(os.environ["DATASET_PATH"])
OUTPUT_PATH = os.environ["OUTPUT_PATH"]


def emit(kind, **data):
    print(json.dumps({"type": kind, "data": data}, ensure_ascii=False), flush=True)


def fail(message):
    emit("error", error=message)
    sys.exit(1)


def read_texts():
    """Returns (train_texts, val_texts)."""
    import pandas as pd
    files = [f for f in sorted(DATASET_PATH.rglob("*"))
             if f.is_file() and f.suffix.lower() in (".parquet", ".csv", ".tsv", ".jsonl", ".txt")
             and not any(p.startswith(".") for p in f.relative_to(DATASET_PATH).parts)]
    if not files:
        fail(f"No parquet/csv/tsv/jsonl/txt files found in {DATASET_PATH}")
    train, val = [], []
    for f in files:
        ext = f.suffix.lower()
        if ext == ".txt":
            rows = [ln.strip() for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        else:
            df = (pd.read_parquet(f) if ext == ".parquet" else
                  pd.read_json(f, lines=True) if ext == ".jsonl" else
                  pd.read_csv(f, sep="\t" if ext == ".tsv" else ","))
            col = TEXT_COLUMN or next((c for c in ("text", "sentence", "content", "review", "document", "body") if c in df.columns), None)
            if col is None:
                strings = [c for c in df.columns if df[c].dtype == object]
                if not strings:
                    continue
                col = max(strings, key=lambda c: df[c].astype(str).str.len().mean())
            rows = [str(v).strip() for v in df[col].dropna().tolist() if str(v).strip()]
        parts = [p.lower() for p in f.relative_to(DATASET_PATH).parts]
        held_out = any(p in ("val", "valid", "validation", "dev", "test") for p in parts[:-1]) or f.stem.lower() in ("val", "valid", "validation", "dev", "test")
        (val if held_out else train).extend(rows)
    if not train:
        train, val = val, []
    random.Random(42).shuffle(train)
    if MAX_ROWS:
        train = train[:MAX_ROWS]
        val = val[: max(MAX_ROWS // 10, 50)]
    if not val:
        cut = max(len(train) // 10, 1)
        train, val = train[cut:], train[:cut]
    return train, val


def load_model():
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(MODEL_PATH), AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
    except Exception as first:
        # Some older BERT checkpoints ship a config.json without model_type.
        if (Path(MODEL_PATH) / "vocab.txt").exists():
            from transformers import BertForMaskedLM, BertTokenizerFast
            return BertTokenizerFast.from_pretrained(MODEL_PATH), BertForMaskedLM.from_pretrained(MODEL_PATH)
        fail(f"This model cannot be loaded for masked language modeling: {first}")


def main():
    import torch
    from transformers import DataCollatorForLanguageModeling

    emit("status", message="Reading text ...")
    train_texts, val_texts = read_texts()
    emit("status", message=f"{len(train_texts)} training texts, {len(val_texts)} validation texts")
    tokenizer, model = load_model()
    if tokenizer.mask_token is None:
        fail("The tokenizer has no mask token. Masked language modeling needs an encoder such as BERT or RoBERTa.")
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=MASK_PROBABILITY)

    def batches(texts, shuffle):
        order = list(range(len(texts)))
        if shuffle:
            random.shuffle(order)
        for i in range(0, len(order), BATCH_SIZE):
            enc = tokenizer([texts[j] for j in order[i:i + BATCH_SIZE]], truncation=True, max_length=MAX_LENGTH)
            feats = [{k: v[n] for k, v in enc.items()} for n in range(len(enc["input_ids"]))]
            yield {k: v.to(device) for k, v in collator(feats).items()}

    def evaluate():
        model.eval()
        torch.manual_seed(0)  # same masks every time, so values are comparable
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in batches(val_texts, False):
                total += float(model(**batch).loss)
                n += 1
        model.train()
        return total / max(n, 1)

    before = evaluate()
    emit("status", message=f"Before training: val loss {before:.4f}, perplexity {math.exp(min(before, 20)):.2f}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = EPOCHS * math.ceil(len(train_texts) / BATCH_SIZE)
    step, val_loss = 0, before
    model.train()
    for epoch in range(1, EPOCHS + 1):
        for batch in batches(train_texts, True):
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % 10 == 0 or step == total_steps:
                emit("progress", epoch=epoch, total_epochs=EPOCHS, step=step, total_steps=total_steps,
                     train_loss=loss.item(), val_loss=val_loss, learning_rate=LEARNING_RATE)
        val_loss = evaluate()
        emit("status", message=f"Epoch {epoch}: val loss {val_loss:.4f}, perplexity {math.exp(min(val_loss, 20)):.2f}")

    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    emit("complete", val_loss_before=before, val_loss=val_loss,
         perplexity_before=math.exp(min(before, 20)), perplexity=math.exp(min(val_loss, 20)),
         train_texts=len(train_texts), steps=step)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
