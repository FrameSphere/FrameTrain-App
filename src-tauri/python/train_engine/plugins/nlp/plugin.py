"""
plugins/nlp/plugin.py
=====================
NLP-Plugin für FrameTrain.

Unterstützt alle HuggingFace Modell-Architekturen:
  - Encoder       (BERT, RoBERTa, XLM-R, DeBERTa, ...)  → Trainer + MLM
  - Encoder-Decoder (T5, mT5, BART, mBART, ...)          → Seq2SeqTrainer
  - Decoder       (GPT-2, LLaMA, Mistral, Phi, ...)      → Trainer + CausalLM

Kompatibilitäts-Fixes:
  - dispatch_batches: Patch für accelerate ≥0.26 / ältere transformers
  - as_target_tokenizer: deprecated in transformers ≥4.35, direkt entfernt
  - eval_strategy vs evaluation_strategy: try/except Fallback
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import TrainingConfig
from core.dataset_manager import find_text_and_target_cols, load_hf_splits
from core.metrics import MetricsCollector
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol


# ============================================================================
# ARCHITEKTUR-ERKENNUNG
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
LORA_MODULES = {
    "encoder":         ["query", "value"],
    "encoder-decoder": ["q", "v"],
    "decoder":         ["q_proj", "v_proj"],
}


def detect_architecture(model_path: str) -> str:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        mt = cfg.model_type.lower().replace("_", "-")
        if mt in ENCODER_TYPES:         return "encoder"
        if mt in ENCODER_DECODER_TYPES: return "encoder-decoder"
        return "decoder"
    except Exception as e:
        MessageProtocol.warning(f"Architektur nicht erkannt ({e}) — verwende Decoder.")
        return "decoder"


# ============================================================================
# ACCELERATE KOMPATIBILITÄTS-PATCH
# ============================================================================

def _patch_accelerate_dispatch_batches():
    """
    Fix für: TypeError: Accelerator.__init__() got an unexpected keyword argument 'dispatch_batches'

    Ursache: transformers (alt) übergibt 'dispatch_batches' an Accelerator,
             aber accelerate ≥0.26 hat diesen Parameter entfernt.

    Lösung: Patch Accelerator.__init__ so dass unbekannte kwargs ignoriert werden.
    """
    try:
        import accelerate
        import inspect
        sig = inspect.signature(accelerate.Accelerator.__init__)
        if "dispatch_batches" not in sig.parameters:
            orig_init = accelerate.Accelerator.__init__

            def _patched_init(self, *args, **kwargs):
                kwargs.pop("dispatch_batches", None)
                kwargs.pop("split_batches", None)       # ebenfalls manchmal betroffen
                orig_init(self, *args, **kwargs)

            accelerate.Accelerator.__init__ = _patched_init
            MessageProtocol.debug("accelerate dispatch_batches patch angewendet")
    except Exception:
        pass  # Falls accelerate nicht installiert / anderer Fehler → ignorieren


# ============================================================================
# RAM-DIAGNOSE
# ============================================================================

def get_system_memory() -> Dict[str, Any]:
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
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return {"total_gb": round(int(out.strip()) / (1024 ** 3), 1),
                    "available_gb": None, "used_gb": None, "percent_used": None}
        except Exception:
            return {"total_gb": None, "available_gb": None,
                    "used_gb": None, "percent_used": None}


def estimate_model_ram_gb(model_path: str, load_in_4bit=False, load_in_8bit=False) -> float:
    try:
        from transformers import AutoConfig
        cfg    = AutoConfig.from_pretrained(model_path)
        h      = getattr(cfg, "hidden_size", 768)
        layers = getattr(cfg, "num_hidden_layers", 12)
        vocab  = getattr(cfg, "vocab_size", 32000)
        params = vocab * h + layers * (4 * h * h + 2 * h * 4 * h)
        bpp    = 0.5 if load_in_4bit else (1.0 if load_in_8bit else 4.0)
        return round(params * bpp / (1024 ** 3) * 1.3, 1)
    except Exception:
        return 2.0


def estimate_token_ram_gb(n_samples: int, seq_length: int) -> float:
    return round(n_samples * seq_length * 3 * 4 / (1024 ** 3), 1)


# ============================================================================
# TRAINING ARGUMENTS (versions-sicher)
# ============================================================================

def _base_args(cfg: TrainingConfig, eval_strat: str, device: str, out_dir: str) -> dict:
    use_fp16      = cfg.fp16 and device == "cuda"
    use_bf16      = cfg.bf16 and device in ("cuda", "cpu")
    save_steps_val = cfg.save_steps if cfg.save_strategy == "steps" else None
    eval_steps_val = cfg.eval_steps  if eval_strat      == "steps" else None

    return dict(
        output_dir=out_dir,
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
        save_steps=save_steps_val,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        eval_steps=eval_steps_val,
        dataloader_num_workers=0,
        seed=cfg.seed,
        report_to="none",
        load_best_model_at_end=(eval_strat != "no"),
        metric_for_best_model="eval_loss" if eval_strat != "no" else None,
        greater_is_better=False,
    )


def build_training_args(cfg: TrainingConfig, eval_strat: str, device: str,
                        out_dir: str, seq2seq: bool = False):
    from transformers import TrainingArguments
    ArgsClass = TrainingArguments
    if seq2seq:
        from transformers import Seq2SeqTrainingArguments
        ArgsClass = Seq2SeqTrainingArguments

    base = _base_args(cfg, eval_strat, device, out_dir)
    if seq2seq:
        base["predict_with_generate"] = True

    # eval_strategy (neu ≥4.38) vs evaluation_strategy (alt)
    eval_candidates = [{"eval_strategy": eval_strat}, {"evaluation_strategy": eval_strat}]
    # Device-Kwargs
    if device == "cpu":
        dev_candidates = [{"use_cpu": True}, {"no_cuda": True}, {}]
    elif device == "mps":
        dev_candidates = [{"use_mps_device": True}, {}]
    else:
        dev_candidates = [{}]

    for ev in eval_candidates:
        for dv in dev_candidates:
            try:
                return ArgsClass(**{**base, **ev, **dv})
            except TypeError:
                continue

    MessageProtocol.warning("TrainingArguments: Kompatibilitätsmodus (minimale Parameter)")
    return ArgsClass(**base)


# ============================================================================
# PROGRESS CALLBACK
# ============================================================================

def make_progress_callback(total_epochs: int, collector: MetricsCollector):
    from transformers import TrainerCallback

    class FTCallback(TrainerCallback):

        def on_epoch_begin(self, args, state, control, **kwargs):
            ep = int(state.epoch or 0) + 1
            MessageProtocol.status("epoch", f"Epoche {ep}/{total_epochs}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            epoch  = int(state.epoch or 0) + 1
            step   = state.global_step or 0
            total  = state.max_steps or 1
            t_loss = float(logs.get("loss") or logs.get("train_loss") or 0.0)
            v_loss = float(logs["eval_loss"]) if "eval_loss" in logs else None
            lr     = float(logs.get("learning_rate") or 0.0)

            MessageProtocol.progress(
                epoch=epoch, total_epochs=total_epochs,
                step=step, total_steps=total,
                train_loss=t_loss, val_loss=v_loss, learning_rate=lr,
            )
            collector.record(epoch, step, {
                "train_loss": t_loss, "val_loss": v_loss, "learning_rate": lr,
            })

        def on_save(self, args, state, control, **kwargs):
            ep = int(state.epoch or 0) + 1
            MessageProtocol.checkpoint(
                step=state.global_step, path=args.output_dir, epoch=ep,
            )

    return FTCallback()


# ============================================================================
# NLP PLUGIN
# ============================================================================

class Plugin(TrainPlugin):
    """HuggingFace NLP-Plugin — Encoder / Encoder-Decoder / Decoder."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._start_time = time.time()
        self.arch: str = "decoder"
        self.device: str = "cpu"
        self.tokenizer = None
        self.model = None
        self.train_tok = None
        self.val_tok = None
        self.trainer = None
        self.metrics = MetricsCollector()
        self._n_train: int = 0
        self._model_ram: float = 0.0

    # ── setup ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        import torch, random
        import numpy as np

        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Accelerate-Compat-Patch VOR allem anderen anwenden
        _patch_accelerate_dispatch_batches()

        if torch.cuda.is_available():
            self.device = "cuda"
            MessageProtocol.status("device", f"GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            MessageProtocol.status("device", "Apple Silicon GPU (MPS)")
        else:
            self.device = "cpu"
            MessageProtocol.status("device", "CPU")

        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.effective_output_dir()).mkdir(parents=True, exist_ok=True)

        mem = get_system_memory()
        self._model_ram = estimate_model_ram_gb(
            self.config.model_path, self.config.load_in_4bit, self.config.load_in_8bit,
        )
        if mem.get("available_gb") and self._model_ram > mem["available_gb"] * 0.8:
            MessageProtocol.warning(
                f"RAM-Warnung: Modell ~{self._model_ram} GB, "
                f"nur {mem['available_gb']} GB verfügbar."
            )

    # ── load_data ─────────────────────────────────────────────────────────

    def load_data(self) -> None:
        cfg = self.config

        MessageProtocol.status("loading", "Erkenne Modell-Architektur...")
        self.arch = detect_architecture(cfg.model_path)
        MessageProtocol.status(
            "loading",
            f"Architektur: {self.arch}  |  Modell: {Path(cfg.model_path).name}"
        )

        from transformers import AutoTokenizer
        MessageProtocol.status("loading", "Lade Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token    = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        train_ds, val_ds, _ = load_hf_splits(cfg.dataset_path)
        self._n_train = len(train_ds)
        MessageProtocol.status("loading", f"Datensatz: {self._n_train:,} Trainingssamples")

        mem = get_system_memory()
        if mem.get("available_gb"):
            tok_ram   = estimate_token_ram_gb(self._n_train, cfg.max_seq_length)
            opt_ram   = round(self._model_ram * 2.0, 1)
            total_est = round(self._model_ram + tok_ram + opt_ram, 1)
            MessageProtocol.status(
                "loading",
                f"RAM-Bedarf: ~{total_est} GB "
                f"(Modell {self._model_ram} + Daten {tok_ram} + Optimizer {opt_ram}) | "
                f"verfügbar: {mem['available_gb']} GB"
            )
            if total_est > mem["available_gb"] * 0.9:
                MessageProtocol.warning(
                    f"RAM-Warnung: {total_est} GB benötigt, {mem['available_gb']} GB verfügbar.\n"
                    "Empfehlung: Batch-Size halbieren oder LoRA aktivieren."
                )

        text_col, target_col = find_text_and_target_cols(train_ds)
        MessageProtocol.status(
            "loading",
            f"Spalten → Text: '{text_col}'"
            + (f"  Ziel: '{target_col}'" if target_col else "  (kein Ziel-Feld)")
        )

        MessageProtocol.status("loading", f"Tokenisiere {self._n_train:,} Samples...")
        self.train_tok = self._tokenize(train_ds, text_col, target_col)

        if val_ds and len(val_ds) > 0:
            MessageProtocol.status("loading", f"Tokenisiere {len(val_ds):,} Validierungs-Samples...")
            self.val_tok = self._tokenize(val_ds, text_col, target_col)
        else:
            MessageProtocol.status("loading", "Kein Validierungsdatensatz — Training ohne Eval")

    def _tokenize(self, dataset, text_col: str, target_col: Optional[str]):
        """
        Tokenisiert den Datensatz.

        WICHTIG: as_target_tokenizer() wurde in transformers ≥4.35 entfernt.
        Für Encoder-Decoder (mT5, T5, BART): Ziel direkt tokenisieren ohne Context Manager.
        """
        cfg  = self.config
        arch = self.arch
        tok  = self.tokenizer

        def tok_fn(examples):
            texts = [str(t) if t is not None else "" for t in examples[text_col]]

            targets = None
            if target_col and target_col in examples:
                targets = []
                for t in examples[target_col]:
                    targets.append(
                        "; ".join(str(x) for x in t) if isinstance(t, list) else str(t or "")
                    )
                if arch != "encoder-decoder":
                    sep = getattr(tok, "sep_token", None) or " → "
                    texts = [f"{i}{sep}{t}" for i, t in zip(texts, targets)]

            enc = tok(texts, truncation=True, max_length=cfg.max_seq_length, padding="max_length")

            if arch == "encoder-decoder":
                if targets is not None:
                    # FIX: as_target_tokenizer() entfernt in transformers ≥4.35
                    # Direkt tokenisieren — funktioniert für T5, mT5, BART, mBART in allen Versionen
                    lenc = tok(
                        targets,
                        truncation=True,
                        max_length=cfg.max_seq_length,
                        padding="max_length",
                        # text_target= wäre neue API, aber not universally supported
                    )
                    # Padding-Token zu -100 (wird vom Loss ignoriert)
                    enc["labels"] = [
                        [(tid if tid != tok.pad_token_id else -100) for tid in ids]
                        for ids in lenc["input_ids"]
                    ]
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

    # ── build_model ───────────────────────────────────────────────────────

    def build_model(self) -> None:
        from transformers import (
            AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        )
        cfg  = self.config
        path = cfg.model_path
        MessageProtocol.status("loading", f"Lade Modell: {Path(path).name}...")

        quant_kwargs: dict = {}
        if cfg.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_kwargs = {
                    "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
                    "device_map": "auto",
                }
            except Exception as e:
                MessageProtocol.warning(f"4-bit Quantisierung nicht verfügbar: {e}")
        elif cfg.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_kwargs = {
                    "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                    "device_map": "auto",
                }
            except Exception as e:
                MessageProtocol.warning(f"8-bit Quantisierung nicht verfügbar: {e}")

        if self.arch == "encoder":
            self.model = AutoModelForMaskedLM.from_pretrained(path, **quant_kwargs)
        elif self.arch == "encoder-decoder":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path, **quant_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, **quant_kwargs)

        n_params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status("loading", f"Modell geladen: {n_params / 1e6:.1f}M Parameter")

        if "device_map" not in quant_kwargs:
            self.model = self.model.to(self.device)
            MessageProtocol.status("loading", f"Modell auf {self.device.upper()} verschoben")

        if cfg.use_lora:
            self._apply_lora()

    def _apply_lora(self) -> None:
        from peft import get_peft_model, LoraConfig, TaskType
        cfg  = self.config
        arch = self.arch
        task_map = {
            "encoder":         TaskType.FEATURE_EXTRACTION,
            "encoder-decoder": TaskType.SEQ_2_SEQ_LM,
            "decoder":         TaskType.CAUSAL_LM,
        }
        modules = cfg.lora_target_modules if cfg.lora_target_modules else LORA_MODULES[arch]
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=modules, bias="none", task_type=task_map[arch],
        )
        self.model = get_peft_model(self.model, lora_cfg)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        MessageProtocol.status(
            "loading", f"LoRA aktiv: {trainable:,} trainierbare Parameter | Module: {modules}"
        )

    # ── train ─────────────────────────────────────────────────────────────

    def train(self) -> None:
        from transformers import (
            Trainer, Seq2SeqTrainer,
            DataCollatorForLanguageModeling, DataCollatorForSeq2Seq,
        )
        cfg = self.config

        if self.arch == "encoder-decoder":
            collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer, model=self.model, padding=True)
        elif self.arch == "encoder":
            collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        else:
            collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False)

        eval_strat = cfg.eval_strategy if self.val_tok is not None else "no"
        out_dir    = cfg.effective_output_dir()
        is_s2s     = (self.arch == "encoder-decoder")

        MessageProtocol.status("loading", "Erstelle TrainingArguments...")
        train_args = build_training_args(cfg, eval_strat, self.device, out_dir, seq2seq=is_s2s)

        TrainerClass = Seq2SeqTrainer if is_s2s else Trainer
        self.trainer = TrainerClass(
            model=self.model,
            args=train_args,
            train_dataset=self.train_tok,
            eval_dataset=self.val_tok,
            tokenizer=self.tokenizer,
            data_collator=collator,
            callbacks=[make_progress_callback(cfg.epochs, self.metrics)],
        )

        self.trainer.train()

    # ── validate ──────────────────────────────────────────────────────────

    def validate(self) -> Dict[str, float]:
        cfg      = self.config
        duration = int(time.time() - self._start_time)
        history  = (self.trainer.state.log_history or []) if self.trainer else []
        total_steps = (self.trainer.state.global_step or 0) if self.trainer else 0

        final_train = next(
            (e["train_loss"] for e in reversed(history) if "train_loss" in e), 0.0
        )
        final_val = next(
            (e["eval_loss"] for e in reversed(history) if "eval_loss" in e), None
        )
        self.metrics.total_epochs = cfg.epochs
        self.metrics.total_steps  = total_steps

        return {
            "final_train_loss":          round(float(final_train), 6),
            "final_val_loss":            round(float(final_val), 6) if final_val is not None else None,
            "total_epochs":              cfg.epochs,
            "total_steps":               total_steps,
            "best_epoch":                self.metrics.best_epoch or 0,
            "training_duration_seconds": duration,
        }

    # ── export ────────────────────────────────────────────────────────────

    def export(self) -> str:
        cfg = self.config
        out = Path(cfg.output_path)
        out.mkdir(parents=True, exist_ok=True)

        if cfg.use_lora and self.model is not None:
            try:
                self.model = self.model.merge_and_unload()
                for p in self.model.parameters():
                    if p.data is not None and not p.is_contiguous():
                        p.data = p.data.contiguous()
                MessageProtocol.status("saving", "LoRA-Gewichte zusammengeführt")
            except Exception as e:
                MessageProtocol.warning(f"LoRA merge fehlgeschlagen: {e}")

        if self.trainer:
            self.trainer.save_model(str(out))
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(out))

        final = self.validate()
        self.metrics.save_with_overrides(str(out), final)

        MessageProtocol.status("saved", f"Modell gespeichert: {out}")
        return str(out)

    def get_info(self) -> Dict[str, Any]:
        return {
            "plugin":          "nlp",
            "architecture":    self.arch,
            "device":          self.device,
            "n_train_samples": self._n_train,
            "model_ram_gb":    self._model_ram,
        }
