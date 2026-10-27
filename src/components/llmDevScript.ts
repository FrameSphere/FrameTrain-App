// Dev-Train-Vorlage fuer Decoder-LLMs (Aufgabe "LLM Fine-Tuning").
//
// Vorher bekam ein LLM wie SmolLM2 die Klassifikations-Vorlage mit
// AutoModelForSequenceClassification — die laeuft auf Chat-Daten nicht.
// Diese Vorlage trainiert LoRA-Adapter (peft) auf Chat- (messages),
// Prompt/Antwort- oder reinen Textdaten und meldet dieselben Events wie die
// anderen Vorlagen.

export function generateCausalLmTrainScript(p: { modelPath: string; datasetPath: string; outputPath: string }): string {
  return `#!/usr/bin/env python3
# FrameTrain - Dev Train Script (LLM, LoRA)
#
# Dieses Template läuft so wie es ist. Ändere es nach Belieben.
# Fortschritt erscheint in FrameTrain, sobald eine JSON-Zeile mit
# {"type": "progress", ...} auf stdout geschrieben wird - genau das macht
# der FrameTrainCallback weiter unten.

import json
import os
from pathlib import Path

import torch
from datasets import load_from_disk, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# -- Pfade (von FrameTrain als ENV-Vars gesetzt) --------------------------
MODEL_PATH   = os.environ.get("MODEL_PATH",   "${p.modelPath}")
DATASET_PATH = os.environ.get("DATASET_PATH", "${p.datasetPath}")
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH",  "${p.outputPath}")

# -- Hyperparameter -------------------------------------------------------
EPOCHS      = 1
BATCH_SIZE  = 2
LR          = 2e-4
MAX_LENGTH  = 512
MAX_STEPS   = 30      # -1 = ganzes Dataset. Klein halten beim Ausprobieren.
LORA_R      = 8
LORA_ALPHA  = 16


def emit(kind: str, **data):
    """Sendet ein Event an FrameTrain (eine JSON-Zeile pro Event)."""
    print(json.dumps({"type": kind, "data": data}), flush=True)


# -- Dataset laden --------------------------------------------------------
def load_frametrain_dataset(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DATASET_PATH existiert nicht: {path}")
    if p.is_dir() and any((p / m).exists() for m in ("dataset_info.json", "state.json", "dataset_dict.json")):
        return load_from_disk(str(p))
    exts = (".json", ".jsonl", ".csv", ".parquet")
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in exts)
    train = [str(f) for f in files if "val" not in f.parts and "test" not in f.parts] or [str(f) for f in files]
    if not train:
        raise RuntimeError(f"Keine Daten-Dateien in {path} gefunden.")
    ext = Path(train[0]).suffix.lower()
    kind = "json" if ext in (".json", ".jsonl") else ext.lstrip(".")
    return load_dataset(kind, data_files={"train": train})


emit("status", stage="loading", message="Dataset wird geladen ...")
dataset = load_frametrain_dataset(DATASET_PATH)
train_ds = dataset["train"]
cols = train_ds.column_names
print("Dataset geladen:", train_ds, flush=True)

# -- Modell & Tokenizer ---------------------------------------------------
emit("status", stage="loading", message="Modell wird geladen ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules="all-linear",
))
model.print_trainable_parameters()
print(f"Modell geladen von {MODEL_PATH} ({device})", flush=True)


def to_text(row) -> str:
    """Eine Zeile als Trainingstext: Chat, Prompt/Antwort oder reiner Text."""
    if "messages" in row and row["messages"]:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return "\\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])
    for q, a in (("prompt", "completion"), ("instruction", "response"), ("question", "answer"), ("input", "output")):
        if q in row and a in row:
            return f"{row[q]}\\n{row[a]}{tokenizer.eos_token}"
    for c in ("text", "content"):
        if c in row:
            return f"{row[c]}{tokenizer.eos_token}"
    raise RuntimeError(f"Keine passende Spalte gefunden. Vorhanden: {cols}")


def tokenize(batch):
    rows = [dict(zip(batch.keys(), vals)) for vals in zip(*batch.values())]
    return tokenizer([to_text(r) for r in rows], truncation=True, max_length=MAX_LENGTH)


train_tok = train_ds.map(tokenize, batched=True, remove_columns=cols)


# -- Fortschritt an FrameTrain melden -------------------------------------
class FrameTrainCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        emit(
            "progress",
            epoch=min(EPOCHS, int(state.epoch or 0) + 1),
            total_epochs=EPOCHS,
            step=state.global_step,
            total_steps=state.max_steps or 0,
            train_loss=logs.get("loss"),
            learning_rate=logs.get("learning_rate", LR),
        )


args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    logging_steps=5,
    save_strategy="no",
    report_to=[],
    disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[FrameTrainCallback()],
)

emit("status", stage="training", message="Training läuft ...")
result = trainer.train()

# -- Speichern ------------------------------------------------------------
# Adapter mit dem Basismodell zusammenführen, damit Tests und Hosting ein
# normales Modell laden können.
merged = model.merge_and_unload()
merged.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"Modell gespeichert unter {OUTPUT_PATH}", flush=True)

emit(
    "complete",
    model_path=OUTPUT_PATH,
    final_metrics={"train_loss": result.training_loss, "total_epochs": EPOCHS},
)
`;
}

// Dev-Test-Vorlage fuer Decoder-LLMs. Vorher bekam auch der Test die
// Klassifikations-Vorlage und brach auf Chat-Daten mit "Text-Spalte nicht
// gefunden. Vorhanden: ['messages']" ab. Diese Vorlage liest dieselben
// Formate wie die Trainingsvorlage, laesst das Modell antworten und
// vergleicht mit der Referenz — ohne Referenz (reiner Text) misst sie die
// Perplexity.
export function generateCausalLmTestScript(p: { modelPath: string; datasetPath: string; outputPath: string }): string {
  return `#!/usr/bin/env python3
# FrameTrain - Dev Test Script (LLM)
#
# Laedt Modell + Test-Split, laesst das Modell antworten und vergleicht mit
# der erwarteten Antwort. Schreibt einen Bericht nach OUTPUT_PATH/results.json.

import json
import math
import os
import random
from pathlib import Path

import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# -- Pfade (von FrameTrain als ENV-Vars gesetzt) --------------------------
MODEL_PATH   = os.environ.get("MODEL_PATH",   "${p.modelPath}")
DATASET_PATH = os.environ.get("DATASET_PATH", "${p.datasetPath}")
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH",  "${p.outputPath}")

MAX_SAMPLES    = 50     # zufaellig gezogen; 0 = alles
MAX_NEW_TOKENS = 64
MAX_LENGTH     = 512

PAIRS = (("prompt", "completion"), ("instruction", "response"), ("question", "answer"), ("input", "output"))


# -- Dataset laden --------------------------------------------------------
def load_rows(path: str):
    # Bevorzugt test/, sonst val/, sonst alles (wie die Trainingsvorlage).
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DATASET_PATH existiert nicht: {path}")
    if p.is_dir() and any((p / m).exists() for m in ("dataset_info.json", "state.json", "dataset_dict.json")):
        ds = load_from_disk(str(p))
        if hasattr(ds, "keys"):
            split = next((s for s in ("test", "validation", "val", "train") if s in ds), list(ds.keys())[0])
            print(f"Split: {split}", flush=True)
            ds = ds[split]
        return [dict(r) for r in ds]
    exts = (".json", ".jsonl", ".csv", ".parquet")
    files = []
    for sub in ("test", "val", "validation", ""):
        d = p / sub if sub else p
        if d.is_dir():
            files = sorted(f for f in d.rglob("*") if f.suffix.lower() in exts)
        if files:
            break
    if not files:
        raise RuntimeError(f"Keine Daten-Dateien in {path} gefunden.")
    print(f"Dateien: {[str(f.relative_to(p)) for f in files]}", flush=True)
    ext = files[0].suffix.lower()
    kind = "json" if ext in (".json", ".jsonl") else ext.lstrip(".")
    return [dict(r) for r in load_dataset(kind, data_files={"test": [str(f) for f in files]})["test"]]


rows = load_rows(DATASET_PATH)
if not rows:
    raise RuntimeError(f"Keine Beispiele in {DATASET_PATH} gefunden.")
if MAX_SAMPLES and len(rows) > MAX_SAMPLES:
    rows = random.Random(42).sample(rows, MAX_SAMPLES)  # zufaellig statt der ersten N
cols = list(rows[0].keys())
print(f"{len(rows)} Beispiele | Spalten: {cols}", flush=True)

# -- Modell & Tokenizer ---------------------------------------------------
print(f"Lade Modell aus: {MODEL_PATH}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
model.eval()
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"Geraet: {device}", flush=True)


def split_row(row):
    """(Prompt-Text, erwartete Antwort) - oder (Text, None) fuer reinen Text."""
    msgs = row.get("messages")
    if msgs:
        # Letzte Assistenten-Antwort ist die Referenz, alles davor der Prompt.
        last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
        context = msgs[:last] if last is not None else msgs
        target = msgs[last]["content"] if last is not None else None
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\\n".join(f"{m['role']}: {m['content']}" for m in context) + "\\nassistant: "
        return prompt, target
    for q, a in PAIRS:
        if q in row and a in row:
            return f"{row[q]}\\n", str(row[a])
    for c in ("text", "content"):
        if c in row:
            return str(row[c]), None
    raise RuntimeError(f"Keine passende Spalte gefunden. Vorhanden: {cols}")


def norm(s: str) -> str:
    return " ".join(str(s).split()).strip(" .!").casefold()


# -- Auswertung -----------------------------------------------------------
predictions, exact, contains, losses = [], 0, 0, []
for i, row in enumerate(rows, 1):
    prompt, target = split_row(row)
    if target is None:
        # Reiner Text: Perplexity statt Antwortvergleich
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            losses.append(float(model(**enc, labels=enc["input_ids"]).loss))
        continue
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                    add_special_tokens=not getattr(tokenizer, "chat_template", None)).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    pred = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    hit = norm(pred) == norm(target)
    exact += hit
    contains += bool(norm(target)) and norm(target) in norm(pred)
    predictions.append({"prompt": prompt[-300:], "predicted": pred, "target": target, "exact": hit})
    if i <= 5:
        print(f"  [{i}] {pred!r}   (erwartet: {target!r})", flush=True)
    if i % 10 == 0:
        print(f"  {i}/{len(rows)} ausgewertet", flush=True)

results = {"n": len(rows), "device": device, "predictions": predictions}
if predictions:
    results["exact_match"] = exact / len(predictions)
    results["contains_match"] = contains / len(predictions)
    print(f"\\nExakte Treffer: {results['exact_match']:.4f}", flush=True)
    print(f"Referenz in Antwort enthalten: {results['contains_match']:.4f}", flush=True)
if losses:
    results["loss"] = sum(losses) / len(losses)
    results["perplexity"] = math.exp(results["loss"])
    print(f"\\nPerplexity: {results['perplexity']:.2f} (Loss {results['loss']:.4f})", flush=True)

# -- Speichern ------------------------------------------------------------
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
with open(os.path.join(OUTPUT_PATH, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Ergebnisse gespeichert: {OUTPUT_PATH}/results.json", flush=True)
`;
}
