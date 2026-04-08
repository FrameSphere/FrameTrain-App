"""
FrameTrain v2 - Universal Training Engine
==========================================
Robust training engine using HuggingFace Trainer + datasets library.
Handles all HF dataset formats and model architectures automatically.

Communication: JSON messages via stdout to Rust backend
"""

import os
import sys
import json
import time
import signal
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# COMMUNICATION PROTOCOL (unchanged — Rust expects this format)
# ============================================================================

class MessageProtocol:
    @staticmethod
    def _send(msg_type: str, data: Dict[str, Any]):
        msg = {"type": msg_type, "timestamp": datetime.now().isoformat(), "data": data}
        print(json.dumps(msg), flush=True)

    @staticmethod
    def progress(epoch, total_epochs, step, total_steps, train_loss,
                 val_loss=None, learning_rate=0.0, metrics=None):
        progress_pct = ((epoch - 1) * total_steps + step) / max(total_epochs * total_steps, 1) * 100
        MessageProtocol._send("progress", {
            "epoch": epoch, "total_epochs": total_epochs,
            "step": step, "total_steps": total_steps,
            "train_loss": train_loss, "val_loss": val_loss,
            "learning_rate": learning_rate,
            "metrics": metrics or {},
            "progress_percent": progress_pct,
        })

    @staticmethod
    def status(status: str, message: str = ""):
        MessageProtocol._send("status", {"status": status, "message": message})

    @staticmethod
    def error(error: str, details: str = ""):
        MessageProtocol._send("error", {"error": error, "details": details})

    @staticmethod
    def warning(message: str):
        MessageProtocol._send("warning", {"message": message})

    @staticmethod
    def complete(model_path: str, metrics: Dict[str, Any]):
        MessageProtocol._send("complete", {
            "model_path": model_path,
            "output_path": model_path,
            "final_metrics": metrics,
        })


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    # Paths
    model_path: str = ""
    dataset_path: str = ""
    output_path: str = ""
    checkpoint_dir: str = ""

    # Training basics
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = -1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.0

    # Optimizer & scheduler
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    scheduler: str = "linear"
    max_grad_norm: float = 1.0

    # Mixed precision
    fp16: bool = False
    bf16: bool = False

    # LoRA
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Data
    max_seq_length: int = 512
    num_workers: int = 0

    # Eval & saving
    eval_steps: int = 500
    eval_strategy: str = "steps"
    save_steps: int = 500
    save_strategy: str = "steps"
    save_total_limit: int = 3
    logging_steps: int = 10

    # Misc
    seed: int = 42
    training_type: str = "fine_tuning"
    task_type: str = "causal_lm"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ============================================================================
# MODEL ARCHITECTURE DETECTION
# ============================================================================

# Encoder-only models (BERT family) → Masked LM
ENCODER_ONLY = {
    "bert", "roberta", "xlm-roberta", "xlm_roberta", "distilbert", "albert",
    "electra", "deberta", "deberta-v2", "camembert", "xlnet", "longformer",
    "bigbird", "rembert", "luke", "ernie", "roformer", "funnel",
}

# Encoder-decoder models (T5 family) → Seq2Seq LM
ENCODER_DECODER = {
    "t5", "mt5", "bart", "mbart", "mbart50", "pegasus", "marian",
    "prophetnet", "led", "longt5", "flan-t5",
}


def detect_architecture(model_path: str) -> str:
    """Returns 'encoder', 'encoder-decoder', or 'decoder'."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        mt = cfg.model_type.lower().replace("_", "-")
        if mt in ENCODER_ONLY:
            return "encoder"
        if mt in ENCODER_DECODER:
            return "encoder-decoder"
        return "decoder"
    except Exception as e:
        MessageProtocol.warning(f"Could not auto-detect architecture ({e}), defaulting to decoder.")
        return "decoder"


# ============================================================================
# ROBUST DATA LOADING
# ============================================================================

SKIP_NAMES = {
    "README.md", "readme.md", "dataset_infos.json", "dataset_dict.json",
    "state.json", "datasetdict.json", ".gitattributes", ".gitignore",
}
SKIP_SUFFIXES = {".md", ".gitattributes", ".gitignore", ".yaml", ".yml", ".lock"}

SPLIT_ALIASES = {
    "train": ["train", "training"],
    "val":   ["val", "validation", "valid", "dev"],
    "test":  ["test", "testing", "eval"],
}


def data_files_in(path: Path) -> List[Path]:
    """Return all data files in path, skipping metadata files."""
    return [
        f for f in path.rglob("*")
        if f.is_file()
        and f.name not in SKIP_NAMES
        and f.suffix.lower() not in SKIP_SUFFIXES
    ]


def load_dir_as_hf_dataset(path: Path):
    """Try loading all data files in a directory as a HuggingFace Dataset."""
    from datasets import Dataset, concatenate_datasets

    files = data_files_in(path)
    if not files:
        return None

    ext_groups: Dict[str, List[str]] = {}
    for f in files:
        ext = f.suffix.lower()
        ext_groups.setdefault(ext, []).append(str(f))

    # Try formats in priority order
    for ext in [".parquet", ".arrow", ".jsonl", ".json", ".csv", ".tsv", ".txt"]:
        group = sorted(ext_groups.get(ext, []))
        if not group:
            continue
        try:
            if ext == ".parquet":
                ds = Dataset.from_parquet(group)
            elif ext == ".arrow":
                parts = [Dataset.from_file(f) for f in group]
                ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            elif ext in (".jsonl", ".json"):
                ds = Dataset.from_json(group)
            elif ext in (".csv", ".tsv"):
                ds = Dataset.from_csv(group)
            elif ext == ".txt":
                rows = []
                for fp in group:
                    with open(fp, encoding="utf-8", errors="replace") as fh:
                        rows.extend({"text": ln.strip()} for ln in fh if ln.strip())
                ds = Dataset.from_list(rows)
            else:
                continue

            MessageProtocol.status(
                "loading",
                f"Loaded {len(ds):,} rows from {len(group)} {ext} file(s) in {path.name}/"
            )
            return ds

        except Exception as e:
            MessageProtocol.warning(f"Could not load {ext} files from {path}: {e}")
            continue

    return None


def load_dataset_from_path(dataset_path: str):
    """
    Load train / val / test splits from dataset_path.
    Tries multiple strategies:
      1. HF DatasetDict (save_to_disk format)
      2. Named split subdirectories (train/, val/, test/)
      3. unused/ directory (FrameTrain pre-split storage)
      4. Root directory scan
    Returns (train_ds, val_ds, test_ds) — val and test may be None.
    """
    from datasets import load_from_disk, DatasetDict

    root = Path(dataset_path)
    MessageProtocol.status("loading", f"Loading dataset from: {root}")

    # Strategy 1: HF DatasetDict saved with save_to_disk()
    try:
        ds = load_from_disk(str(root))
        if isinstance(ds, DatasetDict):
            train = ds.get("train")
            val   = ds.get("validation") or ds.get("val") or ds.get("dev")
            test  = ds.get("test")
            if train is not None:
                MessageProtocol.status("loading", f"Loaded DatasetDict splits: {list(ds.keys())}")
                return train, val, test
    except Exception:
        pass

    # Strategy 2 & 3: Named subdirectories + unused/
    def find_split(split_name: str):
        aliases = SPLIT_ALIASES[split_name]
        if split_name == "train":
            aliases = aliases + ["unused"]  # FrameTrain fallback
        for alias in aliases:
            p = root / alias
            if p.exists():
                ds = load_dir_as_hf_dataset(p)
                if ds is not None:
                    if alias == "unused":
                        MessageProtocol.status("loading", "Using 'unused/' as training data")
                    return ds
        return None

    train_ds = find_split("train")
    val_ds   = find_split("val")
    test_ds  = find_split("test")

    # Strategy 4: Root directory scan
    if train_ds is None:
        MessageProtocol.status("loading", "No split dirs found — scanning root directory...")
        train_ds = load_dir_as_hf_dataset(root)

    if train_ds is None:
        exts = {f.suffix.lower() for f in data_files_in(root)}
        raise ValueError(
            f"No training data found in {root}\n"
            f"Files found with extensions: {exts or 'none'}\n"
            f"Supported: .parquet, .arrow, .jsonl, .json, .csv, .tsv, .txt\n"
            f"Expected directory layout: train/, val/, test/ or files in root"
        )

    return train_ds, val_ds, test_ds


# ============================================================================
# TEXT COLUMN DETECTION
# ============================================================================

TEXT_PRIORITY = [
    "text", "content", "document", "abstract", "body", "passage",
    "sentence", "article", "context", "question", "input", "src",
    "source_text", "input_text", "premise", "title",
]
TARGET_PRIORITY = [
    "keyphrases", "extractive_keyphrases", "abstractive_keyphrases",
    "keywords", "tags", "summary", "target", "output", "tgt",
    "target_text", "answer", "answers", "label", "labels", "response",
]


def find_columns(dataset) -> Tuple[str, Optional[str]]:
    """Find (text_col, target_col) in dataset."""
    cols = dataset.column_names

    text_col = next((c for c in TEXT_PRIORITY if c in cols), None)
    if text_col is None:
        # Fall back: first column with string values
        for c in cols:
            sample = dataset[0].get(c)
            if isinstance(sample, str) and len(sample) > 5:
                text_col = c
                break
    if text_col is None:
        raise ValueError(
            f"No text column found in dataset.\n"
            f"Available columns: {cols}\n"
            f"Known text columns: {TEXT_PRIORITY}"
        )

    target_col = next((c for c in TARGET_PRIORITY if c in cols), None)
    return text_col, target_col


# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize_dataset(dataset, tokenizer, config: TrainingConfig, arch: str,
                     text_col: str, target_col: Optional[str]):
    """Tokenize a HF Dataset for the given architecture."""

    def tokenize_fn(examples):
        texts = [str(t) if t is not None else "" for t in examples[text_col]]

        # For non-encoder models: if target exists, append it (seq2seq-style causal)
        if target_col and target_col in examples and arch == "decoder":
            sep = getattr(tokenizer, "sep_token", None) or " → "
            targets = []
            for t in examples[target_col]:
                if isinstance(t, list):
                    targets.append("; ".join(str(x) for x in t))
                elif t is not None:
                    targets.append(str(t))
                else:
                    targets.append("")
            texts = [f"{inp}{sep}{tgt}" for inp, tgt in zip(texts, targets)]

        enc = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )

        # Causal LM + Masked LM: labels = input_ids (masking for MLM done by DataCollator)
        if arch != "encoder-decoder":
            enc["labels"] = enc["input_ids"].copy()

        return enc

    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    remove = [c for c in dataset.column_names if c not in keep]

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=remove,
        desc="Tokenizing",
    )

    fmt_cols = [c for c in ["input_ids", "attention_mask", "token_type_ids", "labels"]
                if c in tokenized.column_names]
    tokenized.set_format("torch", columns=fmt_cols)
    return tokenized


# ============================================================================
# HUGGINGFACE TRAINER CALLBACK → MessageProtocol bridge
# ============================================================================

def make_progress_callback(total_epochs: int):
    from transformers import TrainerCallback

    class FrameTrainCallback(TrainerCallback):
        def __init__(self):
            self.current_epoch = 0
            self.total_steps_per_epoch = 1
            self.start_time = time.time()

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.current_epoch = int(state.epoch or 0) + 1
            MessageProtocol.status("epoch", f"Epoch {self.current_epoch}/{total_epochs}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            epoch = int(state.epoch or 0) + 1
            step  = state.global_step or 0
            total = state.max_steps or 1

            train_loss = logs.get("loss", logs.get("train_loss", 0.0))
            val_loss   = logs.get("eval_loss")
            lr         = logs.get("learning_rate", 0.0)

            MessageProtocol.progress(
                epoch=epoch,
                total_epochs=total_epochs,
                step=step,
                total_steps=total,
                train_loss=float(train_loss) if train_loss else 0.0,
                val_loss=float(val_loss) if val_loss else None,
                learning_rate=float(lr) if lr else 0.0,
            )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                val_loss = metrics.get("eval_loss")
                if val_loss is not None:
                    MessageProtocol.status("eval", f"Val loss: {val_loss:.4f}")

        def on_save(self, args, state, control, **kwargs):
            MessageProtocol.status("checkpoint", f"Checkpoint saved at step {state.global_step}")

    return FrameTrainCallback()


# ============================================================================
# MAIN TRAINING ENGINE
# ============================================================================

class TrainingEngine:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.is_stopped = False
        signal.signal(signal.SIGINT,  self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, *_):
        MessageProtocol.status("stopping", "Training stopped by user")
        self.is_stopped = True

    def run(self):
        start_time = time.time()

        try:
            import torch
            from transformers import AutoTokenizer, TrainingArguments, Trainer
            from transformers import (
                AutoModelForMaskedLM,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
            )

            cfg = self.config

            # ── 1. Detect device ──────────────────────────────────────────
            if torch.cuda.is_available():
                device = "cuda"
                MessageProtocol.status("device", f"GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                MessageProtocol.status("device", "Apple Silicon GPU (MPS)")
            else:
                device = "cpu"
                MessageProtocol.status("device", "CPU")

            # ── 2. Detect model architecture ──────────────────────────────
            MessageProtocol.status("loading", "Detecting model architecture...")
            arch = detect_architecture(cfg.model_path)
            MessageProtocol.status("loading", f"Architecture: {arch}")

            # ── 3. Load tokenizer ─────────────────────────────────────────
            MessageProtocol.status("loading", "Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # ── 4. Load dataset ───────────────────────────────────────────
            MessageProtocol.status("loading", "Loading dataset...")
            train_ds, val_ds, test_ds = load_dataset_from_path(cfg.dataset_path)

            # ── 5. Find text columns ──────────────────────────────────────
            text_col, target_col = find_columns(train_ds)
            MessageProtocol.status(
                "loading",
                f"Text column: '{text_col}'"
                + (f", target column: '{target_col}'" if target_col else "")
            )

            # ── 6. Tokenize ───────────────────────────────────────────────
            MessageProtocol.status("loading", f"Tokenizing {len(train_ds):,} training samples...")
            train_tok = tokenize_dataset(train_ds, tokenizer, cfg, arch, text_col, target_col)

            val_tok = None
            if val_ds is not None and len(val_ds) > 0:
                MessageProtocol.status("loading", f"Tokenizing {len(val_ds):,} validation samples...")
                val_tok = tokenize_dataset(val_ds, tokenizer, cfg, arch, text_col, target_col)
            else:
                MessageProtocol.status("loading", "No validation data — training without evaluation")

            # ── 7. Load model ─────────────────────────────────────────────
            MessageProtocol.status("loading", f"Loading model from {cfg.model_path}...")

            model_kwargs = {}
            if cfg.load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                model_kwargs["device_map"] = "auto"
            elif cfg.load_in_8bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["device_map"] = "auto"

            if arch == "encoder":
                model = AutoModelForMaskedLM.from_pretrained(cfg.model_path, **model_kwargs)
            elif arch == "encoder-decoder":
                model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_path, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **model_kwargs)

            MessageProtocol.status("loading", f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

            # ── 8. Apply LoRA ─────────────────────────────────────────────
            if cfg.use_lora:
                from peft import get_peft_model, LoraConfig, TaskType
                task_map = {
                    "encoder":         TaskType.FEATURE_EXTRACTION,
                    "encoder-decoder": TaskType.SEQ_2_SEQ_LM,
                    "decoder":         TaskType.CAUSAL_LM,
                }
                lora_cfg = LoraConfig(
                    r=cfg.lora_r,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=cfg.lora_target_modules or None,
                    bias="none",
                    task_type=task_map[arch],
                )
                model = get_peft_model(model, lora_cfg)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                MessageProtocol.status("loading", f"LoRA applied — {trainable:,} trainable params")

            # ── 9. Data collator ──────────────────────────────────────────
            if arch == "encoder":
                from transformers import DataCollatorForLanguageModeling
                collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
                )
            elif arch == "encoder-decoder":
                from transformers import DataCollatorForSeq2Seq
                collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
            else:
                from transformers import DataCollatorForLanguageModeling
                collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            # ── 10. Training arguments ────────────────────────────────────
            eval_strat = cfg.eval_strategy if val_tok is not None else "no"
            save_strat = cfg.save_strategy

            # fp16 not supported on MPS
            use_fp16 = cfg.fp16 and device == "cuda"
            use_bf16 = cfg.bf16 and device in ("cuda", "cpu")

            train_args = TrainingArguments(
                output_dir=cfg.checkpoint_dir or cfg.output_path,
                num_train_epochs=cfg.epochs,
                per_device_train_batch_size=cfg.batch_size,
                per_device_eval_batch_size=cfg.batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                max_steps=cfg.max_steps if cfg.max_steps > 0 else -1,
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                warmup_steps=cfg.warmup_steps,
                warmup_ratio=cfg.warmup_ratio,
                lr_scheduler_type=cfg.scheduler,
                optim="adamw_torch",
                adam_beta1=cfg.adam_beta1,
                adam_beta2=cfg.adam_beta2,
                adam_epsilon=cfg.adam_epsilon,
                max_grad_norm=cfg.max_grad_norm,
                fp16=use_fp16,
                bf16=use_bf16,
                evaluation_strategy=eval_strat,
                eval_steps=cfg.eval_steps if eval_strat == "steps" else None,
                save_strategy=save_strat,
                save_steps=cfg.save_steps if save_strat == "steps" else None,
                save_total_limit=cfg.save_total_limit,
                logging_steps=cfg.logging_steps,
                dataloader_num_workers=0,  # Safer on macOS
                seed=cfg.seed,
                report_to="none",  # Disable wandb/tensorboard
                no_cuda=(device == "cpu"),
                use_mps_device=(device == "mps"),
                load_best_model_at_end=(val_tok is not None),
                metric_for_best_model="eval_loss" if val_tok is not None else None,
                greater_is_better=False,
            )

            # ── 11. Trainer ───────────────────────────────────────────────
            callback = make_progress_callback(cfg.epochs)

            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=train_tok,
                eval_dataset=val_tok,
                tokenizer=tokenizer,
                data_collator=collator,
                callbacks=[callback],
            )

            MessageProtocol.status("training", "Training started...")
            trainer.train()

            if self.is_stopped:
                MessageProtocol.status("stopped", "Training stopped by user")
                return

            # ── 12. Save final model ──────────────────────────────────────
            MessageProtocol.status("saving", f"Saving model to {cfg.output_path}...")
            out = Path(cfg.output_path)
            out.mkdir(parents=True, exist_ok=True)

            # Merge LoRA weights before saving
            if cfg.use_lora:
                try:
                    model = model.merge_and_unload()
                    # Make contiguous
                    for p in model.parameters():
                        if p.data is not None and not p.is_contiguous():
                            p.data = p.data.contiguous()
                    MessageProtocol.status("saving", "LoRA weights merged")
                except Exception as e:
                    MessageProtocol.warning(f"Could not merge LoRA: {e}")

            trainer.save_model(str(out))
            tokenizer.save_pretrained(str(out))

            # ── 13. Collect & save metrics ────────────────────────────────
            duration = int(time.time() - start_time)
            history  = trainer.state.log_history

            final_train_loss = next(
                (e["train_loss"] for e in reversed(history) if "train_loss" in e), 0.0
            )
            final_val_loss = next(
                (e["eval_loss"] for e in reversed(history) if "eval_loss" in e), None
            )

            metrics = {
                "final_train_loss":          final_train_loss,
                "final_val_loss":            final_val_loss,
                "total_epochs":              cfg.epochs,
                "total_steps":               trainer.state.global_step,
                "training_duration_seconds": duration,
                "best_val_loss":             final_val_loss,
            }

            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
            MessageProtocol.status("saved", "Metrics saved")

            MessageProtocol.complete(str(out), metrics)

        except ImportError as e:
            pkg = str(e)
            MessageProtocol.error(
                "Missing Python package",
                f"{pkg}\n\nInstall with:\n  pip install transformers datasets torch peft"
            )
            sys.exit(1)

        except Exception as e:
            MessageProtocol.error("Training engine failed", traceback.format_exc())
            sys.exit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FrameTrain Universal Training Engine")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = json.load(f)

    config = TrainingConfig.from_dict(config_dict)

    try:
        import torch
    except ImportError:
        MessageProtocol.error(
            "PyTorch not installed",
            "Install with: pip install torch transformers datasets"
        )
        sys.exit(1)

    engine = TrainingEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
