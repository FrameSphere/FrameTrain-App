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
# COMMUNICATION PROTOCOL
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
            "learning_rate": learning_rate,
            "metrics": metrics or {},
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

def get_system_memory() -> Dict[str, float]:
    """Returns memory info in GB."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_gb":     round(vm.total     / (1024 ** 3), 1),
            "available_gb": round(vm.available / (1024 ** 3), 1),
            "used_gb":      round(vm.used      / (1024 ** 3), 1),
            "percent_used": vm.percent,
        }
    except ImportError:
        # Fallback: read /proc/meminfo on Linux or sysctl on macOS
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            total = int(out.strip()) / (1024 ** 3)
            return {"total_gb": round(total, 1), "available_gb": None,
                    "used_gb": None, "percent_used": None}
        except Exception:
            return {"total_gb": None, "available_gb": None,
                    "used_gb": None, "percent_used": None}


def estimate_model_ram_gb(model_path: str, load_in_4bit: bool = False,
                           load_in_8bit: bool = False) -> float:
    """Rough estimate of how much RAM loading the model needs."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        params = getattr(cfg, "num_parameters", None)
        if params is None:
            # Estimate from hidden_size + num_layers heuristic
            h = getattr(cfg, "hidden_size", 768)
            layers = getattr(cfg, "num_hidden_layers", 12)
            vocab = getattr(cfg, "vocab_size", 32000)
            params = vocab * h + layers * (4 * h * h + 2 * h * 4 * h)
        bytes_per_param = 0.5 if load_in_4bit else (1.0 if load_in_8bit else 4.0)
        return round(params * bytes_per_param / (1024 ** 3) * 1.3, 1)  # 1.3× overhead
    except Exception:
        return 2.0  # Safe default


def estimate_tokenized_ram_gb(n_samples: int, seq_length: int, n_tensors: int = 3) -> float:
    """Estimate RAM for a fully materialized tokenized dataset.
    n_tensors: typically 3 (input_ids, attention_mask, labels) or 4 with token_type_ids.
    With dynamic padding the actual usage is ~40-60% lower, but this is the safe upper bound.
    """
    return round(n_samples * seq_length * n_tensors * 4 / (1024 ** 3), 1)


def build_oom_message(cfg, mem: Dict, model_ram_est: float, n_samples: int) -> str:
    """Build a human-readable OOM explanation with concrete recommendations."""
    tokenized_ram = estimate_tokenized_ram_gb(n_samples, cfg.max_seq_length, 3)
    optimizer_ram = round(model_ram_est * 2.0, 1)   # AdamW: 2 momentum copies
    total_est = round(model_ram_est + tokenized_ram + optimizer_ram, 1)

    lines = ["\U0001f4be Kein ausreichender RAM für dieses Training.\n"]
    if mem.get("total_gb"):
        lines.append(f"Dein System:       {mem['total_gb']} GB RAM gesamt")
    if mem.get("available_gb"):
        lines.append(f"Verfügbar:         {mem['available_gb']} GB RAM")
    lines.append(f"Modell:           ~{model_ram_est} GB")
    lines.append(f"Tokenisierte Daten:~{tokenized_ram} GB  "
                 f"({n_samples:,} Samples × {cfg.max_seq_length} Tokens × 3 Tensoren × 4 Bytes)")
    lines.append(f"Optimizer (AdamW): ~{optimizer_ram} GB  (2× Modellgröße)")
    lines.append(f"Gesamt geschätzt:  ~{total_est} GB")
    lines.append("")
    lines.append("── Was kannst du tun? ──────────────────────────────")
    lines.append("")

    recommendations = []

    if cfg.batch_size > 1:
        new_bs = max(1, cfg.batch_size // 2)
        recommendations.append(
            f"1. Batch-Size halbieren: {cfg.batch_size} → {new_bs}\n"
            f"   (halbiert RAM-Verbrauch beim Training ohne Qualitätsverlust)"
        )

    if cfg.max_seq_length > 128:
        new_seq = cfg.max_seq_length // 2
        recommendations.append(
            f"2. Max. Sequenzlänge halbieren: {cfg.max_seq_length} → {new_seq}\n"
            f"   (RAM-Verbrauch sinkt quadratisch mit Attention)"
        )

    if not cfg.use_lora:
        recommendations.append(
            "3. LoRA aktivieren\n"
            "   (trainiert nur ~1–5% der Parameter → drastisch weniger RAM)"
        )

    if not cfg.load_in_4bit and not cfg.load_in_8bit:
        recommendations.append(
            "4. 4-bit Quantisierung aktivieren (QLoRA)\n"
            "   (Modell braucht ~4× weniger RAM beim Laden)"
        )

    if n_samples > 10000:
        recommendations.append(
            f"5. Datensatz halbieren\n"
            f"   Dein Datensatz hat {n_samples:,} Samples. Teile ihn über\n"
            f"   'Datensatz → In zwei Hälften teilen' in FrameTrain auf\n"
            f"   und trainiere mit der kleineren Hälfte."
        )

    if mem.get("total_gb") and mem["total_gb"] < 8:
        recommendations.append(
            "6. Mehr RAM nutzen\n"
            "   FrameTrain hat keinen RAM-Limit. Schließe andere Apps,\n"
            "   um mehr RAM freizugeben, bevor du das Training startest."
        )

    lines.extend(recommendations)
    return "\n".join(lines)


def is_oom_error(exc: Exception) -> bool:
    """Check if an exception is an Out-of-Memory error."""
    msg = str(exc).lower()
    oom_keywords = [
        "out of memory", "cannot allocate memory",
        "memoryerror", "memory error", "oom",
        "allocation failed", "killed", "mps backend out of memory",
        "cannot allocate", "not enough memory",
    ]
    return isinstance(exc, MemoryError) or any(k in msg for k in oom_keywords)


# ============================================================================
# CONFIGURATION
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
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    max_seq_length: int = 512
    num_workers: int = 0
    eval_steps: int = 500
    eval_strategy: str = "steps"
    save_steps: int = 500
    save_strategy: str = "steps"
    save_total_limit: int = 3
    logging_steps: int = 10
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

ENCODER_ONLY = {
    "bert", "roberta", "xlm-roberta", "xlm_roberta", "distilbert", "albert",
    "electra", "deberta", "deberta-v2", "camembert", "xlnet", "longformer",
    "bigbird", "rembert", "luke", "ernie", "roformer", "funnel",
}
ENCODER_DECODER = {
    "t5", "mt5", "bart", "mbart", "mbart50", "pegasus", "marian",
    "prophetnet", "led", "longt5",
}


def detect_architecture(model_path: str) -> str:
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
        MessageProtocol.warning(f"Architektur nicht erkannt ({e}), nutze Decoder.")
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
    return [
        f for f in path.rglob("*")
        if f.is_file()
        and f.name not in SKIP_NAMES
        and f.suffix.lower() not in SKIP_SUFFIXES
    ]


def load_dir_as_hf_dataset(path: Path):
    from datasets import Dataset, concatenate_datasets

    files = data_files_in(path)
    if not files:
        return None

    ext_groups: Dict[str, List[str]] = {}
    for f in files:
        ext = f.suffix.lower()
        ext_groups.setdefault(ext, []).append(str(f))

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
                f"Geladen: {len(ds):,} Zeilen aus {len(group)} {ext}-Datei(en) [{path.name}/]"
            )
            return ds
        except Exception as e:
            MessageProtocol.warning(f"Konnte {ext}-Dateien aus {path} nicht laden: {e}")
            continue
    return None


def load_dataset_from_path(dataset_path: str):
    from datasets import load_from_disk, DatasetDict
    root = Path(dataset_path)
    MessageProtocol.status("loading", f"Lade Datensatz aus: {root}")

    # Strategy 1: HF DatasetDict (save_to_disk)
    try:
        ds = load_from_disk(str(root))
        if isinstance(ds, DatasetDict):
            train = ds.get("train")
            val   = ds.get("validation") or ds.get("val") or ds.get("dev")
            test  = ds.get("test")
            if train is not None:
                MessageProtocol.status("loading", f"DatasetDict geladen: {list(ds.keys())}")
                return train, val, test
    except Exception:
        pass

    # Strategy 2: Named subdirectories + unused fallback
    def find_split(split_name: str):
        aliases = SPLIT_ALIASES[split_name]
        if split_name == "train":
            aliases = aliases + ["unused"]
        for alias in aliases:
            p = root / alias
            if p.exists():
                ds = load_dir_as_hf_dataset(p)
                if ds is not None:
                    if alias == "unused":
                        MessageProtocol.status("loading", "Nutze 'unused/' als Trainingsdaten")
                    return ds
        return None

    train_ds = find_split("train")
    val_ds   = find_split("val")
    test_ds  = find_split("test")

    # Strategy 3: Root directory
    if train_ds is None:
        MessageProtocol.status("loading", "Keine Split-Ordner — scanne Root-Verzeichnis...")
        train_ds = load_dir_as_hf_dataset(root)

    if train_ds is None:
        exts = {f.suffix.lower() for f in data_files_in(root)}
        raise ValueError(
            f"Keine Trainingsdaten gefunden in: {root}\n"
            f"Gefundene Dateiendungen: {exts or 'keine'}\n"
            f"Unterstützt: .parquet, .arrow, .jsonl, .json, .csv, .tsv, .txt\n"
            f"Erwartete Ordnerstruktur: train/, val/, test/ oder Dateien im Root"
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
    cols = dataset.column_names
    text_col = next((c for c in TEXT_PRIORITY if c in cols), None)
    if text_col is None:
        for c in cols:
            sample = dataset[0].get(c)
            if isinstance(sample, str) and len(sample) > 5:
                text_col = c
                break
    if text_col is None:
        raise ValueError(
            f"Keine Text-Spalte gefunden.\n"
            f"Verfügbare Spalten: {cols}\n"
            f"Bekannte Text-Spalten: {TEXT_PRIORITY}"
        )
    target_col = next((c for c in TARGET_PRIORITY if c in cols), None)
    return text_col, target_col


# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize_dataset(dataset, tokenizer, config: TrainingConfig, arch: str,
                     text_col: str, target_col: Optional[str]):
    def tokenize_fn(examples):
        texts = [str(t) if t is not None else "" for t in examples[text_col]]
        if target_col and target_col in examples and arch in ("decoder", "encoder"):
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

        # Dynamic padding: truncate to max_length but do NOT pad per sample.
        # Padding happens per-batch inside the DataCollator — saves 50-70% RAM.
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
        )
        if arch != "encoder-decoder":
            enc["labels"] = enc["input_ids"].copy()
        return enc

    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    remove = [c for c in dataset.column_names if c not in keep]
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=remove, desc="Tokenizing")
    # NOTE: no set_format("torch") here — DataCollatorWithPaddingAndLabels handles conversion.
    return tokenized


class DataCollatorWithPaddingAndLabels:
    """
    Custom collator that:
    - Pads input_ids / attention_mask / token_type_ids via tokenizer.pad()
    - Pads labels separately with -100 (ignored in loss) to the same max length
    - Returns proper torch tensors for all fields
    """
    def __init__(self, tokenizer, padding=True):
        self.tokenizer = tokenizer
        self.padding   = padding

    def __call__(self, features):
        import torch

        # Pull labels out before the tokenizer sees the batch
        labels_raw = [f.pop("labels", None) for f in features]

        # Pad the tokenizer fields (input_ids, attention_mask, token_type_ids)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels with -100 to match the padded input length
        if labels_raw[0] is not None:
            max_len = batch["input_ids"].shape[1]  # length after tokenizer padding
            padded = [
                list(lbl) + [-100] * (max_len - len(lbl))
                for lbl in labels_raw
            ]
            batch["labels"] = torch.tensor(padded, dtype=torch.long)

        return batch


# ============================================================================
# PROGRESS CALLBACK
# ============================================================================

def make_progress_callback(total_epochs: int):
    from transformers import TrainerCallback

    class FrameTrainCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            ep = int(state.epoch or 0) + 1
            MessageProtocol.status("epoch", f"Epoche {ep}/{total_epochs}")

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
                epoch=epoch, total_epochs=total_epochs,
                step=step, total_steps=total,
                train_loss=float(train_loss) if train_loss else 0.0,
                val_loss=float(val_loss) if val_loss else None,
                learning_rate=float(lr) if lr else 0.0,
            )

        def on_save(self, args, state, control, **kwargs):
            MessageProtocol.status("checkpoint", f"Checkpoint bei Schritt {state.global_step}")

    return FrameTrainCallback()


# ============================================================================
# MAIN TRAINING ENGINE
# ============================================================================

class TrainingEngine:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.is_stopped = False
        self._n_train_samples = 0
        self._model_ram_est = 0.0
        signal.signal(signal.SIGINT,  self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, *_):
        MessageProtocol.status("stopping", "Training durch User gestoppt")
        self.is_stopped = True

    def _handle_exception(self, exc: Exception):
        """Classify exception and emit appropriate error message."""
        tb = traceback.format_exc()

        if is_oom_error(exc):
            mem = get_system_memory()
            msg = build_oom_message(
                self.config, mem,
                self._model_ram_est,
                self._n_train_samples
            )
            MessageProtocol.error("RAM-Fehler: Nicht genug Arbeitsspeicher", msg)
            return

        if isinstance(exc, ImportError) or "No module named" in str(exc):
            MessageProtocol.error(
                "Fehlendes Python-Paket",
                f"{exc}\n\nInstalliere mit:\n  pip install transformers datasets torch peft psutil"
            )
            return

        # Generic — but still include key info
        MessageProtocol.error("Training Engine Fehler", tb)

    def run(self):
        start_time = time.time()
        cfg = self.config

        try:
            import torch
            from transformers import (
                AutoTokenizer, TrainingArguments, Trainer,
                AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
            )

            # ── Device ───────────────────────────────────────────────────
            if torch.cuda.is_available():
                device = "cuda"
                MessageProtocol.status("device", f"GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                MessageProtocol.status("device", "Apple Silicon GPU (MPS)")
            else:
                device = "cpu"
                MessageProtocol.status("device", "CPU")

            # ── Pre-flight RAM check (model only — dataset not loaded yet) ──
            mem = get_system_memory()
            self._model_ram_est = estimate_model_ram_gb(
                cfg.model_path, cfg.load_in_4bit, cfg.load_in_8bit
            )
            if mem.get("available_gb") and self._model_ram_est > mem["available_gb"] * 0.8:
                MessageProtocol.warning(
                    f"Wenig RAM: {mem['available_gb']} GB frei, "
                    f"Modell braucht ca. {self._model_ram_est} GB. "
                    f"Training könnte fehlschlagen."
                )

            # ── Architecture ──────────────────────────────────────────────
            MessageProtocol.status("loading", "Erkenne Modell-Architektur...")
            arch = detect_architecture(cfg.model_path)
            MessageProtocol.status("loading", f"Architektur: {arch}")

            # ── Tokenizer ─────────────────────────────────────────────────
            MessageProtocol.status("loading", "Lade Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # ── Dataset ───────────────────────────────────────────────────
            MessageProtocol.status("loading", "Lade Datensatz...")
            train_ds, val_ds, test_ds = load_dataset_from_path(cfg.dataset_path)
            self._n_train_samples = len(train_ds)

            # ── Pre-flight: full RAM estimate after dataset size is known ──
            if mem.get("available_gb"):
                tok_ram   = estimate_tokenized_ram_gb(self._n_train_samples, cfg.max_seq_length, 3)
                opt_ram   = round(self._model_ram_est * 2.0, 1)
                total_est = round(self._model_ram_est + tok_ram + opt_ram, 1)
                avail     = mem["available_gb"]
                MessageProtocol.status(
                    "loading",
                    f"RAM-Bedarf: ~{total_est} GB "
                    f"(Modell {self._model_ram_est} + Daten {tok_ram} + Optimizer {opt_ram}), "
                    f"verfügbar: {avail} GB"
                )
                if total_est > avail * 0.90:
                    warning_msg = build_oom_message(
                        cfg, mem, self._model_ram_est, self._n_train_samples
                    )
                    MessageProtocol.warning(
                        f"RAM-Warnung: ~{total_est} GB benötigt, nur {avail} GB verfügbar.\n"
                        f"{warning_msg}"
                    )

            # ── Columns ───────────────────────────────────────────────────
            text_col, target_col = find_columns(train_ds)
            MessageProtocol.status(
                "loading",
                f"Text-Spalte: '{text_col}'"
                + (f", Ziel-Spalte: '{target_col}'" if target_col else "")
            )

            # ── Tokenize ──────────────────────────────────────────────────
            MessageProtocol.status("loading", f"Tokenizing {len(train_ds):,} Trainings-Samples...")
            train_tok = tokenize_dataset(train_ds, tokenizer, cfg, arch, text_col, target_col)

            val_tok = None
            if val_ds is not None and len(val_ds) > 0:
                MessageProtocol.status("loading", f"Tokenizing {len(val_ds):,} Validierungs-Samples...")
                val_tok = tokenize_dataset(val_ds, tokenizer, cfg, arch, text_col, target_col)
            else:
                MessageProtocol.status("loading", "Kein Validierungsdatensatz — Training ohne Eval")

            # ── Model ─────────────────────────────────────────────────────
            MessageProtocol.status("loading", f"Lade Modell aus {cfg.model_path}...")
            model_kwargs = {}
            if cfg.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                    model_kwargs["device_map"] = "auto"
                except Exception as e:
                    MessageProtocol.warning(f"4-bit nicht verfügbar: {e}. Lade ohne Quantisierung.")
            elif cfg.load_in_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    model_kwargs["device_map"] = "auto"
                except Exception as e:
                    MessageProtocol.warning(f"8-bit nicht verfügbar: {e}. Lade ohne Quantisierung.")

            if arch == "encoder":
                model = AutoModelForMaskedLM.from_pretrained(cfg.model_path, **model_kwargs)
            elif arch == "encoder-decoder":
                model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_path, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **model_kwargs)

            n_params = sum(p.numel() for p in model.parameters())
            MessageProtocol.status("loading", f"Modell geladen ({n_params:,} Parameter)")

            # ── LoRA ──────────────────────────────────────────────────────
            if cfg.use_lora:
                from peft import get_peft_model, LoraConfig, TaskType
                task_map = {
                    "encoder":         TaskType.FEATURE_EXTRACTION,
                    "encoder-decoder": TaskType.SEQ_2_SEQ_LM,
                    "decoder":         TaskType.CAUSAL_LM,
                }
                # Auto-select correct LoRA target modules based on architecture.
                # Encoder models (BERT/RoBERTa/xlm-roberta) use 'query'/'value',
                # decoder models (GPT/Llama) use 'q_proj'/'v_proj'.
                default_decoder_modules = ["q_proj", "v_proj"]
                if (not cfg.lora_target_modules or
                        cfg.lora_target_modules == default_decoder_modules) and arch == "encoder":
                    resolved_modules = ["query", "value"]
                    MessageProtocol.status(
                        "loading",
                        f"LoRA: auto-korrigierte Module auf {resolved_modules} "
                        f"(Encoder-Modell erkannt, nicht {default_decoder_modules})"
                    )
                else:
                    resolved_modules = cfg.lora_target_modules or None
                lora_cfg = LoraConfig(
                    r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=resolved_modules,
                    bias="none", task_type=task_map[arch],
                )
                model = get_peft_model(model, lora_cfg)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                MessageProtocol.status("loading", f"LoRA angewendet — {trainable:,} trainierbare Parameter")

            # ── Data Collator (dynamic padding per batch = less RAM) ───────
            if arch == "encoder":
                # Custom collator: pads input fields via tokenizer AND pads labels with -100.
                # DataCollatorWithPadding alone does NOT handle variable-length labels.
                collator = DataCollatorWithPaddingAndLabels(tokenizer=tokenizer, padding=True)
            elif arch == "encoder-decoder":
                from transformers import DataCollatorForSeq2Seq
                collator = DataCollatorForSeq2Seq(
                    tokenizer=tokenizer, model=model, padding=True, pad_to_multiple_of=8
                )
            else:
                from transformers import DataCollatorForLanguageModeling
                collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            # ── Training Arguments ────────────────────────────────────────
            eval_strat = cfg.eval_strategy if val_tok is not None else "no"
            use_fp16   = cfg.fp16 and device == "cuda"
            use_bf16   = cfg.bf16 and device in ("cuda", "cpu")

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
                save_strategy=cfg.save_strategy,
                save_steps=cfg.save_steps if cfg.save_strategy == "steps" else None,
                save_total_limit=cfg.save_total_limit,
                logging_steps=cfg.logging_steps,
                dataloader_num_workers=0,
                seed=cfg.seed,
                report_to="none",
                no_cuda=(device == "cpu"),
                use_mps_device=(device == "mps"),
                load_best_model_at_end=(val_tok is not None),
                metric_for_best_model="eval_loss" if val_tok is not None else None,
                greater_is_better=False,
            )

            # ── Trainer ───────────────────────────────────────────────────
            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=train_tok,
                eval_dataset=val_tok,
                tokenizer=tokenizer,
                data_collator=collator,
                callbacks=[make_progress_callback(cfg.epochs)],
            )

            MessageProtocol.status("training", "Training gestartet...")
            trainer.train()

            if self.is_stopped:
                MessageProtocol.status("stopped", "Training durch User gestoppt")
                return

            # ── Save ──────────────────────────────────────────────────────
            MessageProtocol.status("saving", f"Speichere Modell nach {cfg.output_path}...")
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
            MessageProtocol.status("saved", "Modell und Metriken gespeichert")
            MessageProtocol.complete(str(out), metrics)

        except Exception as exc:
            self._handle_exception(exc)
            sys.exit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FrameTrain Universal Training Engine")
    parser.add_argument("--config", required=True, help="Pfad zur config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = json.load(f)

    try:
        import torch
    except ImportError:
        MessageProtocol.error(
            "PyTorch nicht installiert",
            "Installiere mit: pip install torch transformers datasets"
        )
        sys.exit(1)

    config = TrainingConfig.from_dict(config_dict)
    TrainingEngine(config).run()


if __name__ == "__main__":
    main()
