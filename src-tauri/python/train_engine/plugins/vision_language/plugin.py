"""
Vision-Language (VLM) mit LoRA
==============================
Bild + Frage -> Antwort: Bildbeschreibung, visuelle Fragen, Texterkennung
(OCR-artig), Formular-Felder. Getestet mit SmolVLM (256M/500M), laeuft ueber
AutoModelForImageTextToText auch fuer Qwen2-VL, PaliGemma, LLaVA und BLIP
(Captioning).

Trainiert wird ein LoRA-Adapter auf den Attention-Projektionen des
Sprachmodells; der Bild-Encoder bleibt eingefroren (plugin_config
train_vision=true nimmt ihn dazu). Der Loss zaehlt nur die Antwort-Tokens.

Datenformate (siehe ft_data.image_text):
    JSONL/CSV mit image + question/prompt + answer/caption
    JSONL mit messages im Chat-Format (Bild-Platzhalter im Nutzer-Zug)
    Bilder + gleichnamige .txt  -> Captioning mit plugin_config prompt
    <antwort>/bild.png          -> Ordnername ist die Antwort

Export: LoRA ins Modell gemergt (Test und Labor laden ohne peft), der Adapter
liegt zusaetzlich unter lora_adapter/.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft

from ft_data.image_text import ImageTextSample, resolve_image_text, text_scores
from ft_data import vlm as vlmlib


def _open_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        pc = config.plugin_config or {}
        self.default_prompt = str(pc.get("prompt", "") or "").strip() or vlmlib.DEFAULT_PROMPT
        self.max_new_tokens = int(pc.get("max_new_tokens", 64) or 64)
        self.eval_generate_samples = int(pc.get("eval_generate_samples", 50) or 0)
        self.eval_before = bool(pc.get("eval_before_training", True))
        self.train_vision = bool(pc.get("train_vision", False))
        split_opt = pc.get("image_splitting", False)
        self.image_splitting = None if split_opt in (None, "") else bool(split_opt)
        self.max_pixels = int(pc.get("max_pixels", 0) or 0)
        self.max_length = int(pc.get("max_length", 0) or 0)

        self.processor = None
        self.model = None
        self.model_type = ""
        self.family = "chat"
        self.train_dataset = None
        self.eval_dataset = None
        self._val_samples: List[ImageTextSample] = []
        self.device_used = "cpu"
        self._trainer = None
        self._start_time = time.time()
        self._last_train_loss = 0.0
        self._last_lr = config.learning_rate
        self.baseline: Dict[str, float] = {}
        self.lora_targets: List[str] = []

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        try:
            import peft  # noqa: F401
        except ImportError:
            from ft_data.deps import missing
            raise missing("peft", what="Vision-Language-Training (LoRA)")
        from transformers import AutoConfig, AutoProcessor

        cfg = AutoConfig.from_pretrained(self.config.model_path)
        self.model_type = str(getattr(cfg, "model_type", "") or "")
        if self.model_type in ("clip", "siglip", "siglip2"):
            MessageProtocol.error(
                "Kein generatives Modell",
                f"{self.model_type} ist ein Embedding-Modell (Bild und Text als Vektoren) und erzeugt keinen Text. "
                "Fuer Bild + Frage -> Antwort ein VLM waehlen, z. B. HuggingFaceTB/SmolVLM-256M-Instruct.")
            return False
        MessageProtocol.status("init", f"Architektur: {self.model_type or 'unbekannt'} | Lade Processor...")
        self.processor = AutoProcessor.from_pretrained(self.config.model_path)
        vlmlib.configure_processor(self.processor, self.image_splitting, self.max_pixels)
        self.family = vlmlib.family(self.model_type, self.processor,
                                    bool(getattr(cfg, "is_encoder_decoder", False)))
        label = {"chat": "Chat-Template", "paligemma": "PaliGemma-Praefix", "encdec": "Encoder-Decoder",
                 "plain": "Text-Praefix (ohne Chat-Template)"}[self.family]
        if not str((self.config.plugin_config or {}).get("prompt", "") or "").strip():
            # Ohne Chat-Template ist der Prompt ein Text-Praefix der Antwort. Ein
            # deutscher Standardsatz davor verdarb BLIP die Bildbeschreibung
            # ("en flag of the republic ..."); BLIP beschreibt ohne Praefix.
            if self.family == "plain":
                self.default_prompt = ""
            elif self.model_type == "florence2":
                self.default_prompt = "<CAPTION>"
        MessageProtocol.status("init", f"✓ Processor geladen | Prompt-Format: {label}")

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def load_data(self) -> None:
        from datasets import Dataset

        splits = resolve_image_text(
            Path(self.config.dataset_path), "vlm", seed=self.config.seed, val_fraction=0.1,
            status=lambda m: MessageProtocol.status("loading_data", m))
        for note in splits.notes:
            MessageProtocol.status("loading_data", note)

        def rows(samples: List[ImageTextSample]):
            return {"image": [str(s.image) for s in samples],
                    "question": [s.prompt or self.default_prompt for s in samples],
                    "answer": [s.answer for s in samples]}

        self.train_dataset = Dataset.from_dict(rows(splits.train))
        eval_ds = Dataset.from_dict(rows(splits.val)) if splits.val else None
        if eval_ds is not None:
            eval_ds = hft.cap_eval_dataset(eval_ds, getattr(self.config, "max_eval_samples", 0), self.config.seed)
        self.eval_dataset = eval_ds
        self._val_samples = [ImageTextSample(Path(r["image"]), r["question"], r["answer"])
                             for r in (eval_ds if eval_ds is not None else [])]
        with_q = sum(1 for s in splits.train if s.prompt)
        questions = {s.prompt for s in splits.train if s.prompt}
        if not (self.config.plugin_config or {}).get("prompt") and len(questions) == 1 and with_q == len(splits.train):
            # Stellen alle Beispiele dieselbe Frage, ist sie der beste Standard fuer
            # Test und Labor, wenn dort keine Frage eingegeben wird.
            self.default_prompt = next(iter(questions))
        MessageProtocol.status(
            "loading_data",
            f"✓ Train: {len(self.train_dataset)} | Val: {len(eval_ds) if eval_ds is not None else 0} | "
            + (f"{with_q} mit eigener Frage" if with_q else f"Standardprompt: '{self.default_prompt}'"),
        )

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText

        MessageProtocol.status("building_model", "Lade Vision-Language-Modell...")
        use_bf16 = self.config.bf16 and torch.cuda.is_available()
        model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_path, dtype=torch.bfloat16 if use_bf16 else torch.float32)
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        if self.config.lora_target_modules:
            self.lora_targets = list(self.config.lora_target_modules)
        else:
            self.lora_targets = vlmlib.find_lora_targets(model, self.train_vision)
        lora = LoraConfig(
            r=int(self.config.lora_r), lora_alpha=int(self.config.lora_alpha),
            lora_dropout=float(self.config.lora_dropout or 0.0),
            target_modules=self.lora_targets, bias="none",
        )
        self.model = get_peft_model(model, lora)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        short = sorted({t.split(".")[-1] for t in self.lora_targets})
        MessageProtocol.status(
            "building_model",
            f"✓ {total/1e6:.0f}M Parameter | LoRA r={self.config.lora_r} auf {len(self.lora_targets)} Schichten "
            f"({', '.join(short)}) | trainierbar: {trainable/1e6:.2f}M ({100*trainable/max(total,1):.2f} %)"
            + (" | Bild-Encoder mit LoRA" if self.train_vision else " | Bild-Encoder eingefroren"),
        )

    # ── Generierung fuer die Auswertung ─────────────────────────────────────
    def _generate_eval(self, tag: str) -> Dict[str, float]:
        samples = self._val_samples[: self.eval_generate_samples] if self.eval_generate_samples > 0 else []
        if not samples:
            return {}
        MessageProtocol.status(
            "validating" if tag == "after" else "training",
            f"Erzeuge Antworten fuer {len(samples)} Val-Beispiele ({'vor' if tag == 'before' else 'nach'} dem Training)...")
        model = self.model
        was_training = model.training
        model.eval()
        preds: List[str] = []
        bs = max(1, min(int(self.config.batch_size), 8))
        try:
            for i in range(0, len(samples), bs):
                chunk = samples[i:i + bs]
                preds.extend(vlmlib.generate(
                    model, self.processor, self.family, [_open_rgb(s.image) for s in chunk],
                    [s.prompt for s in chunk], self.max_new_tokens))
        finally:
            if was_training:
                model.train()
        scores = text_scores(preds, [s.answer for s in samples])
        examples = "; ".join(f"'{p}' (Soll: '{s.answer}')" for p, s in list(zip(preds, samples))[:3])
        MessageProtocol.status(
            "validating" if tag == "after" else "training",
            f"{'Vorher' if tag == 'before' else 'Nachher'}: exact match {scores.get('exact_match', 0):.2%}, "
            f"ROUGE-L {scores.get('rougeL', 0):.3f} | Beispiele: {examples}")
        if tag == "after":
            self._examples = [{"question": s.prompt, "expected": s.answer, "predicted": p, "image": str(s.image)}
                              for p, s in zip(preds, samples)]
        return scores

    # ── 4. Training ─────────────────────────────────────────────────────────
    def train(self) -> None:
        from transformers import Trainer, TrainerCallback, TrainingArguments

        self.device_used = hft.device_name()
        processor, fam, max_len = self.processor, self.family, self.max_length

        def collate(batch):
            return vlmlib.collate(
                processor, fam, [_open_rgb(r["image"]) for r in batch],
                [r["question"] for r in batch], [r["answer"] for r in batch], max_len)

        if self.eval_before and self._val_samples:
            import torch
            self.model.to(torch.device(self.device_used))
            self.baseline = self._generate_eval("before")

        total_steps = max(
            (len(self.train_dataset) // max(self.config.batch_size, 1)) * max(self.config.epochs, 1), 1)
        if int(self.config.max_steps) > 0:
            total_steps = int(self.config.max_steps)
        eval_strategy = self.config.eval_strategy if self.eval_dataset is not None else "no"
        args = hft.build_training_arguments(
            self.config, self.config.effective_output_dir(), TrainingArguments,
            remove_unused_columns=False, eval_strategy=eval_strategy,
            # Label-Smoothing auf Antwort-Tokens verfaelscht Antworten wie "rot".
            label_smoothing_factor=0.0,
        )
        self._trainer = Trainer(
            model=self.model, args=args,
            train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
            data_collator=collate,
            callbacks=[hft.progress_callback(TrainerCallback, self, total_steps)],
        )
        MessageProtocol.status("training", f"Training gestartet | Geraet: {self.device_used.upper()}")
        self._start_time = time.time()
        self._trainer.train()
        MessageProtocol.status("training", "Training abgeschlossen")

    # ── 5. Validierung ──────────────────────────────────────────────────────
    def validate(self) -> Dict[str, Any]:
        MessageProtocol.status("validating", "Finale Validierung...")
        result = self._trainer.evaluate() if self.eval_dataset is not None else {}
        metrics = hft.final_metrics(self, self._trainer, result, self._start_time,
                                    architecture=self.model_type, num_labels=0)
        # Der Trainer meldet zum Schluss "train_loss" als Mittel ueber den ganzen
        # Lauf (inkl. der hohen Startwerte). Final ist der letzte Schritt-Loss.
        step_losses = [h["loss"] for h in self._trainer.state.log_history if "loss" in h]
        if step_losses:
            metrics["final_train_loss"] = float(step_losses[-1])
            metrics["first_train_loss"] = float(step_losses[0])
        scores = self._generate_eval("after")
        metrics.update(scores)
        # "accuracy" = exakter Treffer, damit die Analyse-Seite eine Trefferquote zeigt.
        if "exact_match" in scores:
            metrics["accuracy"] = scores["exact_match"]
        for k, v in self.baseline.items():
            metrics[f"baseline_{k}"] = v
        for k in ("f1", "precision", "recall", "num_labels"):
            metrics.pop(k, None)
        return metrics

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        import torch

        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out / "lora_adapter"))
        MessageProtocol.status("export", "LoRA-Adapter gespeichert (lora_adapter/). Merge ins Modell...")
        merged = self.model.merge_and_unload()
        merged.to(torch.float32)
        merged.save_pretrained(str(out))
        self.processor.save_pretrained(str(out))
        info = {
            "default_prompt": self.default_prompt,
            "model_type": self.model_type,
            "family": self.family,
            "max_new_tokens": self.max_new_tokens,
            "image_splitting": self.image_splitting,
            "max_pixels": self.max_pixels,
            "base_model_path": str(Path(self.config.model_path).resolve()),
            "lora_r": int(self.config.lora_r),
            "lora_alpha": int(self.config.lora_alpha),
            "lora_targets": sorted({t.split(".")[-1] for t in self.lora_targets}),
        }
        (out / vlmlib.VLM_INFO_FILE).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        examples = getattr(self, "_examples", None)
        if examples:
            (out / "val_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
        MessageProtocol.status("export", f"Modell gespeichert (gemergt, ohne peft ladbar): {out}")
        return str(out)
