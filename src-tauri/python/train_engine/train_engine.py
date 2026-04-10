"""
FrameTrain - Training Engine
============================
Simple, robust training engine for all HuggingFace model types.

Architecture:
  encoder       → Trainer + MLM/padding collator
  encoder-decoder → Seq2SeqTrainer
  decoder       → Trainer + causal LM collator

Communication: JSON messages via stdout to Rust backend.
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
# MESSAGE PROTOCOL
# ============================================================================

class MessageProtocol:
    @staticmethod
    def _send(msg_type: str, data: Dict[str, Any]):
        msg = {"type": msg_type, "timestamp": datetime.now().isoformat(), "data": data}
        print(json.dumps(msg), flush=True)

    @staticmethod
    def progress(epoch, total_epochs, step, total_steps, train_loss,
                 val_loss=None, learning_rate=0.0, metrics=None):
        pct = ((epoch - 1) * total_steps + step) / max(total_epochs * total_steps, 1) * 100
        MessageProtocol._send("progress", {
            "epoch": epoch, "total_epochs": total_epochs,
            "step": step, "total_steps": total_steps,
            "train_loss": train_loss, "val_loss": val_loss,
            "learning_rate": learning_rate, "metrics": metrics or {},
            "progress_percent": pct,
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
# RAM DIAGNOSTICS
# ============================================================================

def get_system_memory() -> Dict[str, Any]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_gb":     round(vm.total     / (1024**3), 1),
            "available_gb": round(vm.available / (1024**3), 1),
            "used_gb":      round(vm.used      / (1024**3), 1),
            "percent_used": vm.percent,
        }
    except ImportError:
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return {"total_gb": round(int(out.strip()) / (1024**3), 1),
                    "available_gb": None, "used_gb": None, "percent_used": None}
        except Exception:
            return {"total_gb": None, "available_gb": None,
                    "used_gb": None, "percent_used": None}


def estimate_model_ram_gb(model_path: str, load_in_4bit=False, load_in_8bit=False) -> float:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        h      = getattr(cfg, "hidden_size", 768)
        layers = getattr(cfg, "num_hidden_layers", 12)
        vocab  = getattr(cfg, "vocab_size", 32000)
        params = vocab * h + layers * (4 * h * h + 2 * h * 4 * h)
        bpp    = 0.5 if load_in_4bit else (1.0 if load_in_8bit else 4.0)
        return round(params * bpp / (1024**3) * 1.3, 1)
    except Exception:
        return 2.0


def estimate_tokenized_ram_gb(n_samples: int, seq_length: int, n_tensors: int = 3) -> float:
    return round(n_samples * seq_length * n_tensors * 4 / (1024**3), 1)


def build_oom_advice(cfg, mem, model_ram, n_samples) -> str:
    tok_ram = estimate_tokenized_ram_gb(n_samples, cfg.max_seq_length)
    opt_ram = round(model_ram * 2.0, 1)
    total   = round(model_ram + tok_ram + opt_ram, 1)
    avail   = mem.get("available_gb", "?")
    lines = [
        f"Geschätzter RAM-Bedarf: ~{total} GB  (Modell {model_ram} + Daten {tok_ram} + Optimizer {opt_ram})",
        f"Verfügbar: {avail} GB",
        "",
        "Empfehlungen:",
        f"  1. Batch-Size halbieren  (aktuell: {cfg.batch_size}  → {max(1, cfg.batch_size//2)})",
        f"  2. Sequenzlänge halbieren (aktuell: {cfg.max_seq_length} → {cfg.max_seq_length//2})",
        "  3. LoRA aktivieren (nur 1–5 % der Parameter trainieren)",
        "  4. Andere Apps schließen",
        "  5. Datensatz weiter aufteilen",
    ]
    return "\n".join(lines)


def is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or any(k in msg for k in [
        "out of memory", "cannot allocate memory", "memoryerror", "oom",
        "allocation failed", "mps backend out of memory", "not enough memory",
    ])


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class TrainingConfig:
    model_path: str = ""
    dataset_path: str = ""
    output_path: str = ""
    checkpoint_dir: str = ""
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = -1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    scheduler: str = "linear"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [])
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    max_seq_length: int = 512
    eval_steps: int = 500
    eval_strategy: str = "steps"
    save_steps: int = 500
    save_strategy: str = "steps"
    save_total_limit: int = 3
    logging_steps: int = 50
    seed: int = 42
    training_type: str = "fine_tuning"
    task_type: str = "auto"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ============================================================================
# ARCHITECTURE DETECTION
# ============================================================================

ENCODER_TYPES = {
    "bert", "roberta", "xlm-roberta", "xlm_roberta", "distilbert", "albert",
    "electra", "deberta", "deberta-v2", "camembert", "xlnet", "longformer",
    "bigbird", "rembert", "luke", "ernie", "roformer", "funnel",
}
ENCODER_DECODER_TYPES = {
    "t5", "mt5", "bart", "mbart", "mbart50", "pegasus", "marian",
    "prophetnet", "led", "longt5",
}

# LoRA modules per architecture family
LORA_MODULES = {
    "encoder":         ["query", "value"],          # BERT/RoBERTa attention layers
    "encoder-decoder": ["q", "v"],                  # T5 attention layers
    "decoder":         ["q_proj", "v_proj"],         # GPT/Llama attention layers
}


def detect_architecture(model_path: str) -> str:
    """Returns 'encoder', 'encoder-decoder', or 'decoder'."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        mt  = cfg.model_type.lower().replace("_", "-")
        if mt in ENCODER_TYPES:         return "encoder"
        if mt in ENCODER_DECODER_TYPES: return "encoder-decoder"
        return "decoder"
    except Exception as e:
        MessageProtocol.warning(f"Architektur nicht erkannt ({e}), nutze Decoder.")
        return "decoder"


# ============================================================================
# DATASET LOADING
# ============================================================================

SKIP_NAMES = {
    "README.md", "readme.md", "dataset_infos.json", "dataset_dict.json",
    "state.json", "datasetdict.json", ".gitattributes", ".gitignore",
}
SKIP_SUFFIXES = {".md", ".gitattributes", ".gitignore", ".yaml", ".yml", ".lock"}

SPLIT_ALIASES = {
    "train": ["train", "training", "unused"],
    "val":   ["val", "validation", "valid", "dev"],
    "test":  ["test", "testing"],
}

TEXT_COLS = [
    "text", "content", "document", "abstract", "body", "passage",
    "sentence", "article", "context", "question", "input", "src",
    "source_text", "input_text", "premise", "hypothesis",
]
TARGET_COLS = [
    "keyphrases", "extractive_keyphrases", "abstractive_keyphrases",
    "keywords", "tags", "summary", "target", "output", "tgt",
    "target_text", "answer", "answers", "label", "labels", "response",
]


def _data_files(path: Path) -> List[Path]:
    return [
        f for f in path.rglob("*")
        if f.is_file()
        and f.name not in SKIP_NAMES
        and f.suffix.lower() not in SKIP_SUFFIXES
    ]


def _load_dir(path: Path):
    """Load a directory as a HuggingFace Dataset. Tries formats in order."""
    from datasets import Dataset, concatenate_datasets

    files = _data_files(path)
    if not files:
        return None

    # Group by extension
    by_ext: Dict[str, List[str]] = {}
    for f in files:
        by_ext.setdefault(f.suffix.lower(), []).append(str(f))

    for ext in [".parquet", ".arrow", ".jsonl", ".json", ".csv", ".tsv", ".txt"]:
        group = sorted(by_ext.get(ext, []))
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
            MessageProtocol.status("loading",
                f"Geladen: {len(ds):,} Zeilen aus {len(group)} {ext}-Datei(en) [{path.name}/]")
            return ds
        except Exception as e:
            MessageProtocol.warning(f"{ext}-Dateien aus {path} nicht lesbar: {e}")
    return None


def load_dataset_splits(dataset_path: str):
    """Returns (train_ds, val_ds, test_ds). train_ds is always non-None."""
    from datasets import load_from_disk, DatasetDict
    root = Path(dataset_path)
    MessageProtocol.status("loading", f"Lade Datensatz: {root}")

    # Strategy 1: HF save_to_disk format
    try:
        ds = load_from_disk(str(root))
        if isinstance(ds, DatasetDict):
            train = ds.get("train")
            val   = ds.get("validation") or ds.get("val") or ds.get("dev")
            test  = ds.get("test")
            if train is not None:
                MessageProtocol.status("loading", f"DatasetDict-Splits: {list(ds.keys())}")
                return train, val, test
    except Exception:
        pass

    # Strategy 2: named subdirectories
    def find_split(key):
        for alias in SPLIT_ALIASES[key]:
            p = root / alias
            if p.exists():
                d = _load_dir(p)
                if d is not None:
                    if alias == "unused":
                        MessageProtocol.status("loading", "Nutze 'unused/' als Trainingsdaten")
                    return d
        return None

    train_ds = find_split("train")
    val_ds   = find_split("val")
    test_ds  = find_split("test")

    # Strategy 3: root directory
    if train_ds is None:
        MessageProtocol.status("loading", "Keine Split-Ordner — scanne Root-Verzeichnis...")
        train_ds = _load_dir(root)

    if train_ds is None:
        exts = {f.suffix.lower() for f in _data_files(root)}
        raise ValueError(
            f"Keine Trainingsdaten in: {root}\n"
            f"Gefundene Endungen: {exts or 'keine'}\n"
            f"Unterstützt: .parquet .arrow .jsonl .json .csv .tsv .txt"
        )

    return train_ds, val_ds, test_ds


def find_text_target_cols(dataset) -> Tuple[str, Optional[str]]:
    cols = dataset.column_names
    text_col = next((c for c in TEXT_COLS if c in cols), None)
    if text_col is None:
        # Fall back: first column with non-empty string values
        for c in cols:
            sample = dataset[0].get(c)
            if isinstance(sample, str) and len(sample) > 5:
                text_col = c
                break
    if text_col is None:
        raise ValueError(
            f"Keine Text-Spalte gefunden.\n"
            f"Verfügbare Spalten: {cols}\n"
            f"Erwartete Spalten: {TEXT_COLS[:6]}..."
        )
    target_col = next((c for c in TARGET_COLS if c in cols), None)
    return text_col, target_col


# ============================================================================
# PROGRESS CALLBACK
# ============================================================================

def make_progress_callback(total_epochs: int):
    from transformers import TrainerCallback

    class FTCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            ep = int(state.epoch or 0) + 1
            MessageProtocol.status("epoch", f"Epoche {ep}/{total_epochs}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            epoch = int(state.epoch or 0) + 1
            step  = state.global_step or 0
            total = state.max_steps or 1
            MessageProtocol.progress(
                epoch=epoch, total_epochs=total_epochs,
                step=step, total_steps=total,
                train_loss=float(logs.get("loss") or logs.get("train_loss") or 0.0),
                val_loss=float(logs["eval_loss"]) if "eval_loss" in logs else None,
                learning_rate=float(logs.get("learning_rate") or 0.0),
            )

        def on_save(self, args, state, control, **kwargs):
            MessageProtocol.status("checkpoint", f"Checkpoint Schritt {state.global_step}")

    return FTCallback()


# ============================================================================
# TRAINING ARGUMENTS — version-safe builder
# ============================================================================

def _make_base_args(cfg: TrainingConfig, eval_strat: str, device: str, output_dir: str) -> dict:
    """Build the kwargs dict for TrainingArguments (without deprecated params)."""
    use_fp16 = cfg.fp16 and device == "cuda"
    use_bf16 = cfg.bf16 and device in ("cuda", "cpu")
    return dict(
        output_dir=output_dir,
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
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps if cfg.save_strategy == "steps" else None,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        dataloader_num_workers=0,
        seed=cfg.seed,
        report_to="none",
        load_best_model_at_end=(eval_strat != "no"),
        metric_for_best_model="eval_loss" if eval_strat != "no" else None,
        greater_is_better=False,
    )


def build_training_args(cfg, eval_strat, device, output_dir, seq2seq=False):
    """
    Build TrainingArguments/Seq2SeqTrainingArguments compatible with multiple
    transformers versions by trying different parameter names via try/except.
    """
    from transformers import TrainingArguments
    if seq2seq:
        from transformers import Seq2SeqTrainingArguments as ArgsClass
    else:
        ArgsClass = TrainingArguments

    base = _make_base_args(cfg, eval_strat, device, output_dir)
    if seq2seq:
        base["predict_with_generate"] = True

    # Candidates for eval strategy param name (new → old)
    eval_candidates = [
        {"eval_strategy": eval_strat},
        {"evaluation_strategy": eval_strat},
    ]
    # Candidates for eval_steps (only when strategy != "no")
    if eval_strat != "no":
        base["eval_steps"] = cfg.eval_steps if eval_strat == "steps" else None

    # Device params: try modern first, then legacy, then nothing
    if device == "cpu":
        device_candidates = [{"use_cpu": True}, {"no_cuda": True}, {}]
    elif device == "mps":
        # use_mps_device was removed in transformers ≥ 4.38 → MPS is auto-detected
        device_candidates = [{"use_mps_device": True}, {}]
    else:
        device_candidates = [{}]

    for eval_kw in eval_candidates:
        for dev_kw in device_candidates:
            try:
                kwargs = {**base, **eval_kw, **dev_kw}
                return ArgsClass(**kwargs)
            except TypeError:
                continue

    # Last resort: no deprecated params at all
    MessageProtocol.warning("TrainingArguments: Nutze minimale Parameter (Kompatibilitätsmodus)")
    return ArgsClass(**base)


# ============================================================================
# MAIN TRAINING ENGINE
# ============================================================================

class TrainingEngine:

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.is_stopped = False
        self._n_train = 0
        self._model_ram = 0.0
        signal.signal(signal.SIGINT,  self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, *_):
        MessageProtocol.status("stopping", "Training gestoppt")
        self.is_stopped = True

    # ── Error handler ──────────────────────────────────────────────────────

    def _handle_exc(self, exc: Exception):
        tb = traceback.format_exc()

        if is_oom(exc):
            mem = get_system_memory()
            advice = build_oom_advice(self.cfg, mem, self._model_ram, self._n_train)
            MessageProtocol.error("RAM-Fehler: Nicht genug Arbeitsspeicher", advice)
            return

        if isinstance(exc, ImportError) or "No module named" in str(exc):
            MessageProtocol.error(
                "Fehlendes Python-Paket",
                f"{exc}\n\nInstalliere mit:\n  pip install transformers datasets torch peft psutil"
            )
            return

        if isinstance(exc, TypeError) and "unexpected keyword argument" in str(exc):
            MessageProtocol.error(
                "API-Inkompatibilität",
                f"{exc}\n\nAktualisiere transformers:\n  pip install --upgrade transformers\n\n{tb}"
            )
            return

        # Generic — full traceback always visible
        MessageProtocol.error(f"Training-Fehler: {type(exc).__name__}: {exc}", tb)

    # ── Tokenization ───────────────────────────────────────────────────────

    def _tokenize(self, dataset, tokenizer, arch: str, text_col: str, target_col: Optional[str]):
        cfg = self.cfg

        def tok_fn(examples):
            texts = [str(t) if t is not None else "" for t in examples[text_col]]

            # Concatenate target if present (input → target for supervised learning)
            if target_col and target_col in examples:
                sep = getattr(tokenizer, "sep_token", None) or " → "
                targets = []
                for t in examples[target_col]:
                    if isinstance(t, list):
                        targets.append("; ".join(str(x) for x in t))
                    else:
                        targets.append(str(t) if t is not None else "")
                if arch != "encoder-decoder":
                    # For encoder/decoder: concatenate as single sequence
                    texts = [f"{inp}{sep}{tgt}" for inp, tgt in zip(texts, targets)]

            enc = tokenizer(
                texts,
                truncation=True,
                max_length=cfg.max_seq_length,
                padding="max_length",   # simple, reliable, avoids custom collator issues
            )

            if arch == "encoder-decoder":
                # For seq2seq: tokenize targets separately as labels
                if target_col and target_col in examples:
                    with tokenizer.as_target_tokenizer():
                        label_enc = tokenizer(
                            targets,
                            truncation=True,
                            max_length=cfg.max_seq_length,
                            padding="max_length",
                        )
                    labels = [
                        [(t if t != tokenizer.pad_token_id else -100) for t in ids]
                        for ids in label_enc["input_ids"]
                    ]
                    enc["labels"] = labels
                else:
                    enc["labels"] = enc["input_ids"].copy()
            else:
                enc["labels"] = enc["input_ids"].copy()

            return enc

        keep   = {"input_ids", "attention_mask", "token_type_ids", "labels"}
        remove = [c for c in dataset.column_names if c not in keep]
        tokenized = dataset.map(tok_fn, batched=True, remove_columns=remove, desc="Tokenizing")
        fmt_cols  = [c for c in ["input_ids", "attention_mask", "token_type_ids", "labels"]
                     if c in tokenized.column_names]
        tokenized.set_format("torch", columns=fmt_cols)
        return tokenized

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_model(self, arch: str, quant_kwargs: dict):
        from transformers import (
            AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM
        )
        path = self.cfg.model_path
        if arch == "encoder":
            return AutoModelForMaskedLM.from_pretrained(path, **quant_kwargs)
        elif arch == "encoder-decoder":
            return AutoModelForSeq2SeqLM.from_pretrained(path, **quant_kwargs)
        else:
            return AutoModelForCausalLM.from_pretrained(path, **quant_kwargs)

    # ── LoRA ───────────────────────────────────────────────────────────────

    def _apply_lora(self, model, arch: str):
        from peft import get_peft_model, LoraConfig, TaskType
        cfg = self.cfg
        task_map = {
            "encoder":         TaskType.FEATURE_EXTRACTION,
            "encoder-decoder": TaskType.SEQ_2_SEQ_LM,
            "decoder":         TaskType.CAUSAL_LM,
        }
        # Pick correct modules for this architecture
        modules = cfg.lora_target_modules if cfg.lora_target_modules else LORA_MODULES[arch]
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=modules,
            bias="none", task_type=task_map[arch],
        )
        model = get_peft_model(model, lora_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        MessageProtocol.status("loading", f"LoRA: {trainable:,} trainierbare Parameter ({modules})")
        return model

    # ── Main ───────────────────────────────────────────────────────────────

    def run(self):
        start = time.time()
        cfg   = self.cfg
        try:
            import torch
            from transformers import (
                AutoTokenizer, Trainer, Seq2SeqTrainer,
                DataCollatorForLanguageModeling, DataCollatorForSeq2Seq,
            )

            # ── Device ──────────────────────────────────────────────────
            if torch.cuda.is_available():
                device = "cuda"
                MessageProtocol.status("device", f"GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                MessageProtocol.status("device", "Apple Silicon GPU (MPS)")
            else:
                device = "cpu"
                MessageProtocol.status("device", "CPU")

            # ── RAM pre-check (model only) ───────────────────────────────
            mem = get_system_memory()
            self._model_ram = estimate_model_ram_gb(
                cfg.model_path, cfg.load_in_4bit, cfg.load_in_8bit)
            if mem.get("available_gb") and self._model_ram > mem["available_gb"] * 0.8:
                MessageProtocol.warning(
                    f"RAM-Warnung: Modell braucht ~{self._model_ram} GB, "
                    f"nur {mem['available_gb']} GB frei."
                )

            # ── Architecture ─────────────────────────────────────────────
            MessageProtocol.status("loading", "Erkenne Modell-Architektur...")
            arch = detect_architecture(cfg.model_path)
            MessageProtocol.status("loading", f"Architektur: {arch}  (Modell: {Path(cfg.model_path).name})")

            # ── Tokenizer ────────────────────────────────────────────────
            MessageProtocol.status("loading", "Lade Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token    = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # ── Dataset ──────────────────────────────────────────────────
            train_ds, val_ds, _ = load_dataset_splits(cfg.dataset_path)
            self._n_train = len(train_ds)
            MessageProtocol.status("loading", f"Datensatz: {self._n_train:,} Trainingssamples")

            # ── RAM check after dataset size known ───────────────────────
            if mem.get("available_gb"):
                tok_ram   = estimate_tokenized_ram_gb(self._n_train, cfg.max_seq_length)
                opt_ram   = round(self._model_ram * 2.0, 1)
                total_est = round(self._model_ram + tok_ram + opt_ram, 1)
                avail     = mem["available_gb"]
                MessageProtocol.status(
                    "loading",
                    f"RAM-Bedarf: ~{total_est} GB  "
                    f"(Modell {self._model_ram} + Daten {tok_ram} + Optimizer {opt_ram}), "
                    f"verfügbar: {avail} GB"
                )
                if total_est > avail * 0.90:
                    advice = build_oom_advice(cfg, mem, self._model_ram, self._n_train)
                    MessageProtocol.warning(f"RAM-Warnung — Training könnte fehlschlagen:\n{advice}")

            # ── Text columns ─────────────────────────────────────────────
            text_col, target_col = find_text_target_cols(train_ds)
            MessageProtocol.status(
                "loading",
                f"Spalten: Text='{text_col}'"
                + (f", Ziel='{target_col}'" if target_col else " (kein Ziel-Feld)")
            )

            # ── Tokenize ─────────────────────────────────────────────────
            MessageProtocol.status("loading", f"Tokenisiere {self._n_train:,} Trainingssamples...")
            train_tok = self._tokenize(train_ds, tokenizer, arch, text_col, target_col)

            val_tok = None
            if val_ds and len(val_ds) > 0:
                MessageProtocol.status("loading", f"Tokenisiere {len(val_ds):,} Validierungssamples...")
                val_tok = self._tokenize(val_ds, tokenizer, arch, text_col, target_col)
            else:
                MessageProtocol.status("loading", "Kein Validierungsdatensatz — Training ohne Eval")

            # ── Model ────────────────────────────────────────────────────
            MessageProtocol.status("loading", f"Lade Modell: {cfg.model_path}...")
            quant_kwargs: dict = {}
            if cfg.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quant_kwargs = {
                        "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
                        "device_map": "auto",
                    }
                except Exception as e:
                    MessageProtocol.warning(f"4-bit nicht verfügbar: {e}")
            elif cfg.load_in_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quant_kwargs = {
                        "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                        "device_map": "auto",
                    }
                except Exception as e:
                    MessageProtocol.warning(f"8-bit nicht verfügbar: {e}")

            model = self._load_model(arch, quant_kwargs)
            n_params = sum(p.numel() for p in model.parameters())
            MessageProtocol.status("loading", f"Modell geladen: {n_params/1e6:.1f}M Parameter")

            # Move model to device (only when not using device_map="auto")
            if "device_map" not in quant_kwargs:
                model = model.to(device)
                MessageProtocol.status("loading", f"Modell auf {device.upper()} verschoben")

            # ── LoRA ─────────────────────────────────────────────────────
            if cfg.use_lora:
                model = self._apply_lora(model, arch)

            # ── Collator ─────────────────────────────────────────────────
            if arch == "encoder-decoder":
                collator = DataCollatorForSeq2Seq(
                    tokenizer=tokenizer, model=model, padding=True)
            elif arch == "encoder":
                # MLM collator for encoder fine-tuning / domain adaptation
                # Wraps our already-padded sequences; mlm_probability=0.15
                collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
            else:
                collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=False)

            # ── Training args ─────────────────────────────────────────────
            eval_strat = (cfg.eval_strategy if val_tok is not None else "no")
            out_dir    = cfg.checkpoint_dir or cfg.output_path
            is_seq2seq = (arch == "encoder-decoder")

            MessageProtocol.status("loading", "Erstelle TrainingArguments...")
            train_args = build_training_args(cfg, eval_strat, device, out_dir, seq2seq=is_seq2seq)

            # ── Trainer ───────────────────────────────────────────────────
            TrainerClass = Seq2SeqTrainer if is_seq2seq else Trainer
            trainer = TrainerClass(
                model=model,
                args=train_args,
                train_dataset=train_tok,
                eval_dataset=val_tok,
                tokenizer=tokenizer,
                data_collator=collator,
                callbacks=[make_progress_callback(cfg.epochs)],
            )

            # ── Train ─────────────────────────────────────────────────────
            MessageProtocol.status("training", "Training gestartet...")
            trainer.train()

            if self.is_stopped:
                MessageProtocol.status("stopped", "Training durch User gestoppt")
                return

            # ── Save ──────────────────────────────────────────────────────
            MessageProtocol.status("saving", f"Speichere nach {cfg.output_path}...")
            out = Path(cfg.output_path)
            out.mkdir(parents=True, exist_ok=True)

            if cfg.use_lora:
                try:
                    model = model.merge_and_unload()
                    for p in model.parameters():
                        if p.data is not None and not p.is_contiguous():
                            p.data = p.data.contiguous()
                    MessageProtocol.status("saving", "LoRA Gewichte zusammengeführt")
                except Exception as e:
                    MessageProtocol.warning(f"LoRA merge fehlgeschlagen: {e}")

            trainer.save_model(str(out))
            tokenizer.save_pretrained(str(out))

            # ── Metrics ───────────────────────────────────────────────────
            duration = int(time.time() - start)
            history  = trainer.state.log_history
            final_train = next(
                (e["train_loss"] for e in reversed(history) if "train_loss" in e), 0.0)
            final_val = next(
                (e["eval_loss"] for e in reversed(history) if "eval_loss" in e), None)
            metrics = {
                "final_train_loss":          final_train,
                "final_val_loss":            final_val,
                "total_epochs":              cfg.epochs,
                "total_steps":               trainer.state.global_step,
                "training_duration_seconds": duration,
            }
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
            MessageProtocol.status("saved", "Modell und Metriken gespeichert")
            MessageProtocol.complete(str(out), metrics)

        except Exception as exc:
            self._handle_exc(exc)
            sys.exit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FrameTrain Training Engine")
    parser.add_argument("--config", required=True, help="Pfad zur config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = json.load(f)

    try:
        import torch
    except ImportError:
        MessageProtocol.error(
            "PyTorch nicht installiert",
            "pip install torch transformers datasets"
        )
        sys.exit(1)

    config = TrainingConfig.from_dict(config_dict)
    TrainingEngine(config).run()


if __name__ == "__main__":
    main()