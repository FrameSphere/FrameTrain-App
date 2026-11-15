"""Fine-tune CLIP on your own image + caption pairs (contrastive learning).

Teaches CLIP your products, style or domain terms, for image search and
zero-shot classification. Works with any CLIP checkpoint in Hugging Face format
(openai/clip-vit-base-patch32, laion/CLIP-ViT-B-32-..., ...).

Data, any of:
  - metadata.csv / metadata.jsonl with file_name and caption (or text)
  - an image next to a .txt file of the same name (photo_1.jpg + photo_1.txt)
A val/ folder is used for validation; otherwise 15 % of the pairs are held out.

Reported: contrastive loss and retrieval recall@1 (image -> caption) on validation.
"""
import json, math, os, random, sys
from pathlib import Path

EPOCHS = 3
BATCH_SIZE = 16          # contrastive learning profits from larger batches
LEARNING_RATE = 1e-5
MAX_PAIRS = 0            # 0 = all
FREEZE_VISION = False    # True: only the text tower learns (less memory)

MODEL_PATH = os.environ["MODEL_PATH"]
DATASET_PATH = Path(os.environ["DATASET_PATH"])
OUTPUT_PATH = os.environ["OUTPUT_PATH"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


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
            name = r.get("file_name") or r.get("image") or r.get("file")
            cap = r.get("caption") or r.get("text") or r.get("prompt")
            if name and cap and (meta.parent / name).exists():
                pairs[str(meta.parent / name)] = str(cap).strip()
    for img in sorted(DATASET_PATH.rglob("*")):
        if img.suffix.lower() in IMAGE_EXTS and str(img) not in pairs and img.with_suffix(".txt").exists():
            cap = img.with_suffix(".txt").read_text(encoding="utf-8", errors="ignore").strip()
            if cap:
                pairs[str(img)] = cap
    items = sorted(pairs.items())
    if len(items) < 4:
        fail("Fewer than 4 image + caption pairs found. Expected metadata.csv (file_name, caption) "
             "or an image with a .txt file of the same name.")
    val = [p for p in items if any(s in Path(p[0]).relative_to(DATASET_PATH).parts for s in ("val", "valid", "validation", "test"))]
    train = [p for p in items if p not in val]
    random.Random(42).shuffle(train)
    if MAX_PAIRS:
        train = train[:MAX_PAIRS]
    if not val:
        cut = max(int(len(train) * 0.15), 2)
        train, val = train[cut:], train[:cut]
    return train, val


def main():
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    train, val = read_pairs()
    emit("status", message=f"{len(train)} training pairs, {len(val)} validation pairs")
    try:
        model = CLIPModel.from_pretrained(MODEL_PATH)
        processor = CLIPProcessor.from_pretrained(MODEL_PATH)
    except Exception as e:
        fail(f"This is not a CLIP model in Hugging Face format: {e}")
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    if FREEZE_VISION:
        for p in model.vision_model.parameters():
            p.requires_grad = False

    def encode(batch):
        images = [Image.open(p).convert("RGB") for p, _ in batch]
        enc = processor(text=[c for _, c in batch], images=images, return_tensors="pt",
                        padding=True, truncation=True, max_length=model.config.text_config.max_position_embeddings)
        return {k: v.to(device) for k, v in enc.items()}

    def evaluate():
        model.eval()
        losses, hits, n = [], 0, 0
        with torch.no_grad():
            for i in range(0, len(val), BATCH_SIZE):
                batch = val[i:i + BATCH_SIZE]
                if len(batch) < 2:
                    continue
                out = model(**encode(batch), return_loss=True)
                losses.append(float(out.loss))
                # identical captions count as a hit for each other
                best = out.logits_per_image.argmax(dim=1).tolist()
                hits += sum(batch[b][1] == batch[j][1] for j, b in enumerate(best))
                n += len(batch)
        model.train()
        return (sum(losses) / max(len(losses), 1)), (hits / max(n, 1))

    val_loss, recall = evaluate()
    before = {"val_loss": val_loss, "recall_at_1": recall}
    emit("status", message=f"Before training: val loss {val_loss:.4f}, recall@1 {recall:.3f}")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.1)
    steps_per_epoch = max(len(train) // BATCH_SIZE, 1)
    total_steps, step = EPOCHS * steps_per_epoch, 0
    model.train()
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train)
        for i in range(0, steps_per_epoch * BATCH_SIZE, BATCH_SIZE):
            batch = train[i:i + BATCH_SIZE]
            if len(batch) < 2:
                continue
            loss = model(**encode(batch), return_loss=True).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            emit("progress", epoch=epoch, total_epochs=EPOCHS, step=step, total_steps=total_steps,
                 train_loss=loss.item(), val_loss=val_loss, learning_rate=LEARNING_RATE)
        val_loss, recall = evaluate()
        emit("status", message=f"Epoch {epoch}: val loss {val_loss:.4f}, recall@1 {recall:.3f}")

    model.save_pretrained(OUTPUT_PATH)
    processor.save_pretrained(OUTPUT_PATH)
    emit("complete", val_loss=val_loss, recall_at_1=recall, val_loss_before=before["val_loss"],
         recall_at_1_before=before["recall_at_1"], train_pairs=len(train), val_pairs=len(val))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
