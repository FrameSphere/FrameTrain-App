"""Whisper fine-tuning with LoRA (small adapter instead of all weights).

Adapts Whisper to your speakers, accent or vocabulary with little memory:
only the LoRA matrices are trained. At the end the adapter is merged into the
model, so the saved version behaves like a normal Whisper model in FrameTrain
(tests, laboratory, hosting).

Data: audio + transcript, either  clip.wav + clip.txt  (same name) or
metadata.csv / metadata.jsonl with file_name and transcription (or text).
A val/ or test/ folder is used for validation; otherwise 10 % is held out.

Reported: training loss and word error rate (WER) before and after.
"""
import json, os, random, re, sys
from pathlib import Path

EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
LORA_R = 16
LORA_ALPHA = 32
LANGUAGE = ""            # e.g. "german"; empty = let Whisper detect
MAX_SECONDS = 30
MAX_CLIPS = 0            # 0 = all

MODEL_PATH = os.environ["MODEL_PATH"]
DATASET_PATH = Path(os.environ["DATASET_PATH"])
OUTPUT_PATH = os.environ["OUTPUT_PATH"]
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def emit(kind, **data):
    print(json.dumps({"type": kind, "data": data}, ensure_ascii=False), flush=True)


def fail(message):
    emit("error", error=message)
    sys.exit(1)


def read_pairs():
    import csv
    pairs = {}
    for meta in sorted(DATASET_PATH.rglob("metadata.*")):
        if meta.suffix.lower() == ".jsonl":
            rows = [json.loads(ln) for ln in meta.read_text(encoding="utf-8").splitlines() if ln.strip()]
        elif meta.suffix.lower() == ".csv":
            with open(meta, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            continue
        for r in rows:
            name = r.get("file_name") or r.get("path") or r.get("audio")
            text = r.get("transcription") or r.get("transcript") or r.get("sentence") or r.get("text")
            if name and text and (meta.parent / name).exists():
                pairs[str(meta.parent / name)] = str(text).strip()
    for audio in sorted(DATASET_PATH.rglob("*")):
        if audio.suffix.lower() in AUDIO_EXTS and str(audio) not in pairs and audio.with_suffix(".txt").exists():
            text = audio.with_suffix(".txt").read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                pairs[str(audio)] = text
    items = sorted(pairs.items())
    if len(items) < 2:
        fail("No audio + transcript pairs found. Expected clip.wav + clip.txt or metadata.csv (file_name, transcription).")
    val = [p for p in items if any(s in Path(p[0]).relative_to(DATASET_PATH).parts for s in ("val", "valid", "validation", "test"))]
    train = [p for p in items if p not in val]
    random.Random(42).shuffle(train)
    if MAX_CLIPS:
        train = train[:MAX_CLIPS]
    if not val:
        cut = max(len(train) // 10, 1)
        train, val = train[cut:], train[:cut]
    return train, val


def words(text):
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def wer(pairs):
    """Word error rate over (reference, hypothesis) pairs."""
    errors = total = 0
    for ref, hyp in pairs:
        r, h = words(ref), words(hyp)
        row = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, row[0] = row[0], i
            for j in range(1, len(h) + 1):
                prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
        errors += row[-1]
        total += len(r)
    return errors / max(total, 1)


def main():
    import librosa
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    train, val = read_pairs()
    emit("status", message=f"{len(train)} training clips, {len(val)} validation clips")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    except Exception as e:
        fail(f"This is not a Whisper model: {e}")
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    gen_kwargs = {"max_new_tokens": 128}
    if LANGUAGE:
        gen_kwargs.update(language=LANGUAGE, task="transcribe")
        processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task="transcribe")

    def features(paths):
        waves = [librosa.load(p, sr=16000, duration=MAX_SECONDS)[0] for p in paths]
        return processor.feature_extractor(waves, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    def transcribe(m):
        m.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(val), BATCH_SIZE):
                batch = val[i:i + BATCH_SIZE]
                ids = m.generate(input_features=features([p for p, _ in batch]), **gen_kwargs)
                texts = processor.batch_decode(ids, skip_special_tokens=True)
                out += [(ref, hyp.strip()) for (_, ref), hyp in zip(batch, texts)]
        m.train()
        return out

    wer_before = wer(transcribe(model))
    emit("status", message=f"Before training: WER {wer_before:.3f}")

    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"], bias="none"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emit("status", message=f"LoRA: {trainable / 1e6:.2f} M trainable parameters")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    steps_per_epoch = max((len(train) + BATCH_SIZE - 1) // BATCH_SIZE, 1)
    total_steps, step = EPOCHS * steps_per_epoch, 0
    model.train()
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train)
        for i in range(0, len(train), BATCH_SIZE):
            batch = train[i:i + BATCH_SIZE]
            labels = processor.tokenizer([t for _, t in batch], return_tensors="pt", padding=True,
                                         truncation=True, max_length=model.config.max_target_positions)
            ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100).to(device)
            loss = model(input_features=features([p for p, _ in batch]), labels=ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            emit("progress", epoch=epoch, total_epochs=EPOCHS, step=step, total_steps=total_steps,
                 train_loss=loss.item(), learning_rate=LEARNING_RATE)
        emit("status", message=f"Epoch {epoch} done")

    merged = model.merge_and_unload()
    results = transcribe(merged)
    wer_after = wer(results)
    emit("status", message=f"After training: WER {wer_after:.3f} (before {wer_before:.3f})")
    merged.save_pretrained(OUTPUT_PATH)
    processor.save_pretrained(OUTPUT_PATH)
    emit("complete", wer_before=wer_before, wer=wer_after, train_clips=len(train), val_clips=len(val),
         trainable_parameters=trainable, example={"reference": results[0][0], "transcript": results[0][1]} if results else None)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
