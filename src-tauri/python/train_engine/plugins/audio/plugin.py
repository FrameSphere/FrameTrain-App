"""
plugins/audio/plugin.py
=======================
Audio-Plugin für FrameTrain — Speech Recognition Fine-Tuning.

Unterstützt:
  - Whisper Fine-Tuning (openai-whisper / transformers)
  - Wav2Vec2 Fine-Tuning (ASR)
  - Auto-detection über model_path

Dataset-Formate:
  - HuggingFace AudioFolder (metadata.csv + audio-Dateien)
  - CSV mit Spalten 'path' und 'text'/'label'

metrics.json Format: flat auf Root-Level (Rust-kompatibel).
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import TrainingConfig
from core.metrics import MetricsCollector
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol


def detect_audio_architecture(model_path: str) -> str:
    p = model_path.lower()
    if any(k in p for k in ["whisper"]):                      return "whisper"
    if any(k in p for k in ["wav2vec", "xlsr", "mms"]):      return "wav2vec2"
    if any(k in p for k in ["hubert"]):                       return "hubert"
    return "auto"


class Plugin(TrainPlugin):
    """Audio / ASR Plugin — Whisper oder Wav2Vec2."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._start_time = time.time()
        self.arch: str = "auto"
        self.device: str = "cpu"
        self.model = None
        self.processor = None
        self.train_ds = None
        self.val_ds = None
        self.trainer = None
        self._task: str = "asr"
        self.metrics = MetricsCollector()

    def setup(self) -> None:
        import torch, random
        import numpy as np

        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

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
        self.arch = detect_audio_architecture(self.config.model_path)
        MessageProtocol.status("loading", f"Audio-Architektur: {self.arch}")

    def load_data(self) -> None:
        root = Path(self.config.dataset_path)
        MessageProtocol.status("loading", "Lade Audio-Datensatz...")

        # Strategie 1: HuggingFace AudioFolder
        try:
            from datasets import load_dataset, Audio
            metadata = root / "metadata.csv"
            if metadata.exists():
                ds = load_dataset("audiofolder", data_dir=str(root), split=None)
                self.train_ds = ds.get("train")
                self.val_ds   = ds.get("validation") or ds.get("test")
                if self.train_ds:
                    SR = 16_000
                    self.train_ds = self.train_ds.cast_column("audio", Audio(sampling_rate=SR))
                    if self.val_ds:
                        self.val_ds = self.val_ds.cast_column("audio", Audio(sampling_rate=SR))
                    MessageProtocol.status("loading", f"AudioFolder: {len(self.train_ds):,} Samples")
                    return
        except Exception as e:
            MessageProtocol.warning(f"AudioFolder nicht geladen: {e}")

        # Strategie 2: CSV
        csv_files = list(root.glob("train*.csv")) + list(root.glob("train/*.csv"))
        if csv_files:
            from datasets import Dataset
            import csv
            rows = []
            with open(csv_files[0], "r") as f:
                for row in csv.DictReader(f):
                    rows.append(row)
            self.train_ds = Dataset.from_list(rows)
            MessageProtocol.status("loading", f"CSV-Dataset: {len(self.train_ds):,} Samples")
            return

        raise ValueError(
            f"Kein Audio-Dataset in: {root}\n"
            "Unterstützt:\n"
            "  1. HuggingFace AudioFolder (metadata.csv + Audio-Dateien)\n"
            "  2. CSV mit Spalten 'path' und 'text'/'label'"
        )

    def build_model(self) -> None:
        path = self.config.model_path
        MessageProtocol.status("loading", f"Lade Audio-Modell: {Path(path).name}...")

        if self.arch == "whisper":
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            self.processor = WhisperProcessor.from_pretrained(path)
            self.model     = WhisperForConditionalGeneration.from_pretrained(path)
            self.model.config.forced_decoder_ids = None
            self.model.config.suppress_tokens    = []
        elif self.arch in ("wav2vec2", "hubert"):
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            self.processor = Wav2Vec2Processor.from_pretrained(path)
            self.model     = Wav2Vec2ForCTC.from_pretrained(path)
        else:
            try:
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                self.processor = AutoProcessor.from_pretrained(path)
                self.model     = AutoModelForSpeechSeq2Seq.from_pretrained(path)
            except Exception:
                from transformers import AutoProcessor, AutoModelForAudioClassification
                self.processor = AutoProcessor.from_pretrained(path)
                self.model     = AutoModelForAudioClassification.from_pretrained(path)
                self._task     = "classification"

        n_params = sum(p.numel() for p in self.model.parameters())
        MessageProtocol.status("loading", f"Modell geladen: {n_params / 1e6:.1f}M Parameter")

    def train(self) -> None:
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

        cfg = self.config
        MessageProtocol.status("training", "Starte Audio-Training...")

        class _CB(TrainerCallback):
            def __init__(self, total, mc):
                self.total = total; self.mc = mc
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs: return
                ep   = int(state.epoch or 0) + 1
                step = state.global_step or 0
                tl   = float(logs.get("loss") or 0.0)
                vl   = float(logs["eval_loss"]) if "eval_loss" in logs else None
                MessageProtocol.progress(
                    epoch=ep, total_epochs=self.total,
                    step=step, total_steps=state.max_steps or 1,
                    train_loss=tl, val_loss=vl,
                )
                self.mc.record(ep, step, {"train_loss": tl, "val_loss": vl})

        train_args = Seq2SeqTrainingArguments(
            output_dir=cfg.effective_output_dir(),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            save_strategy="epoch",
            logging_steps=cfg.logging_steps,
            predict_with_generate=(self._task == "asr"),
            report_to="none",
            seed=cfg.seed,
        )

        feat_extractor = (self.processor.feature_extractor
                          if hasattr(self.processor, "feature_extractor")
                          else self.processor)

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            tokenizer=feat_extractor,
            callbacks=[_CB(cfg.epochs, self.metrics)],
        )
        self.trainer.train()

        self.metrics.total_epochs = cfg.epochs
        self.metrics.total_steps  = self.trainer.state.global_step or 0

    def validate(self) -> Dict[str, float]:
        duration = int(time.time() - self._start_time)
        history  = self.trainer.state.log_history if self.trainer else []
        total_steps = self.trainer.state.global_step if self.trainer else 0

        final_train = next((e["train_loss"] for e in reversed(history) if "train_loss" in e), 0.0)
        final_val   = next((e["eval_loss"]  for e in reversed(history) if "eval_loss"  in e), None)

        return self.metrics.final_metrics({
            "final_train_loss":          float(final_train),
            "final_val_loss":            float(final_val) if final_val is not None else None,
            "total_epochs":              self.config.epochs,
            "total_steps":               total_steps,
            "training_duration_seconds": duration,
            "architecture":              self.arch,
        })

    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)

        if self.trainer:
            self.trainer.save_model(str(out))
        if self.processor:
            self.processor.save_pretrained(str(out))

        meta = {"architecture": self.arch, "task": self._task, "model": self.config.model_path}
        (out / "config.json").write_text(json.dumps(meta, indent=2))

        # metrics.json flat (Rust-kompatibel)
        final = self.validate()
        self.metrics.save_with_overrides(str(out), final)

        MessageProtocol.status("saved", f"Audio-Modell gespeichert: {out}")
        return str(out)
