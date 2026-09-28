"""
plugins/causal_lm/plugin.py
===========================
Fine-Tuning von Decoder-LLMs (Llama, Qwen, Mistral, Gemma, Phi, SmolLM, GPT-2 …).

Standard ist LoRA: vollstaendiges Fine-Tuning eines 7B-Modells braucht mit
AdamW rund 16 Byte pro Parameter (>100 GB) — lokal ausgeschlossen. LoRA
trainiert nur kleine Zusatzmatrizen (<1 % der Parameter).

Zwei Backends, gleiche Daten, gleiche Ausgabe (HF-Format + LoRA-Adapter):
  - "torch": transformers + peft. Laeuft ueberall (CUDA, Apple MPS, CPU).
    QLoRA (4/8 bit) nur auf NVIDIA — bitsandbytes rechnet nicht auf MPS.
  - "mlx":  mlx-lm auf Apple Silicon. Deutlich schneller auf dem Mac und das
    einzige lokale QLoRA dort (4-bit-Quantisierung in MLX).
"auto" nimmt MLX auf Apple Silicon, wenn mlx-lm installiert ist und die
Architektur kennt, sonst torch.

plugin_config:
  backend           "auto" | "torch" | "mlx"
  system_prompt     wird jedem Beispiel ohne eigene System-Nachricht vorangestellt
  prompt_column / response_column   andere Spaltennamen als die erkannten
  eval_samples      wie viele Val-Beispiele nach dem Training generiert und
                    mit der Referenz verglichen werden (0 = aus)
  eval_baseline     dieselben Beispiele auch VOR dem Training generieren
  max_new_tokens    Laenge der Antworten in dieser Auswertung
  lora_layers       MLX: wie viele Bloecke (von hinten) LoRA bekommen, -1 = alle
  export_gguf       zusaetzlich eine GGUF-Datei fuer Ollama / LM Studio schreiben
  gguf_type         q8_0 (Standard, halbe Groesse) | f16 | bf16 | f32
  dpo_beta          DPO (Daten mit chosen/rejected): Staerke der Praeferenz, Standard 0.1
  dpo_sft_weight    DPO: Anteil SFT-Loss auf der guten Antwort (RPO), 0 = reines DPO
"""
from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol
from core import hf_training as hft

from ft_data import llm as D  # noqa: E402  (gemeinsam mit dem Test-Plugin)
from ft_data import deps as D_deps  # noqa: E402

SUPPORTED_ARCHITECTURES = {
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe",
    "gemma", "gemma2", "gemma3", "gemma3_text", "phi", "phi3", "phimoe",
    "gpt2", "gpt_neo", "gpt_neox", "gptj", "falcon", "bloom", "opt", "mpt",
    "stablelm", "olmo", "olmo2", "granite", "cohere", "starcoder2", "smollm3",
}


class _StopTraining(Exception):
    """Abbruch aus dem MLX-Callback — mlx-lm kennt kein Stop-Flag."""


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _mlx_supports(model_path: Path) -> Optional[str]:
    """None, wenn MLX das Modell trainieren kann — sonst der Grund."""
    if not _is_apple_silicon():
        return "kein Apple Silicon"
    try:
        from mlx_lm.utils import _get_classes, load_config
    except ImportError:
        return "mlx-lm ist nicht installiert (Einstellungen → Python-Pakete → „LLM Fine-Tuning“)"
    try:
        _get_classes(load_config(model_path))
    except Exception as exc:
        return f"mlx-lm kennt die Architektur nicht ({exc})"
    return None


def _mlx_dataset(items, batch_size: int):
    """Tokenisierte Beispiele als mlx-lm-Dataset.

    mlx-lm erwartet process() -> (tokens, offset) und maskiert den Loss vor
    `offset`. Es kennt nur EINEN Praefix: bei mehreren Antworten in einem Chat
    zaehlt ab der ersten Antwort alles (der torch-Weg maskiert jede Frage).
    Weniger Beispiele als batch_size lehnt mlx-lm ab — dann wird wiederholt.
    """
    from mlx_lm.tuner.datasets import CacheDataset

    items = list(items)
    while len(items) < batch_size:
        items = items + items[: batch_size - len(items)]

    class _Masked:
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            return i

        def process(self, i):
            return items[i].input_ids, items[i].first_target

    ds = CacheDataset(_Masked())
    # itemlen() sortiert nach len(data[idx]) — hier die Tokenzahl.
    ds.itemlen = lambda idx: len(items[idx].input_ids)
    return ds


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        pc = config.plugin_config or {}
        self.pc = pc
        self.model_path = Path(config.model_path)
        self.model_type = ""
        self.model_cfg: Dict[str, Any] = {}
        self.backend = str(pc.get("backend", "auto") or "auto").lower()
        self.eval_samples = int(pc.get("eval_samples", 10) or 0)
        self.eval_baseline = bool(pc.get("eval_baseline", True))
        self.max_new_tokens = int(pc.get("max_new_tokens", 128) or 128)
        self.data: Optional[D.LoadedData] = None
        self.tokenizer = None
        self.model = None
        self.template_source = "model"
        self.device_used = "cpu"
        self.quant_bits = 0
        self.trainable_params = 0
        self.total_params = 0
        self.baseline: Dict[str, float] = {}
        self.after: Dict[str, float] = {}
        self.samples: List[Dict[str, Any]] = []
        self.train_dataset = None
        self.eval_dataset = None
        self._trainer = None
        self._start_time = time.time()
        self._last_train_loss: float = 0.0
        self._last_lr = config.learning_rate
        self._last_val_loss: Optional[float] = None
        self._baseline_val_loss: Optional[float] = None
        self._steps_done = 0
        # MLX
        self._mlx_cfg: Dict[str, Any] = {}
        self._mlx_train = None
        self._mlx_val = None

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> bool:
        cfg_file = self.model_path / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(
                f"Keine config.json in {self.model_path} — kein gueltiges HuggingFace-Modell.")
        self.model_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        self.model_type = str(self.model_cfg.get("model_type", "")).lower()
        if self.model_type not in SUPPORTED_ARCHITECTURES:
            MessageProtocol.warning(
                f"Architektur '{self.model_type}' ist nicht in der geprueften Liste — "
                "Training wird trotzdem versucht (AutoModelForCausalLM).")

        want_quant = 4 if self.config.load_in_4bit else 8 if self.config.load_in_8bit else 0
        # Schon quantisierte MLX-Modelle (mlx-community/...-4bit) kann nur MLX laden.
        mlx_quantized = "quantization" in self.model_cfg
        if mlx_quantized and self.backend == "torch":
            raise ValueError(
                "Dieses Modell liegt MLX-quantisiert vor (z. B. mlx-community/…-4bit) und laesst sich "
                "nur mit dem Backend 'mlx' trainieren. Backend auf 'auto' oder 'mlx' stellen oder die "
                "unquantisierte Originalversion importieren.")
        self.backend = self._choose_backend(want_quant)
        if mlx_quantized and self.backend != "mlx":
            raise ValueError(f"MLX-quantisiertes Modell, aber MLX ist nicht nutzbar: {_mlx_supports(self.model_path)}.")
        if want_quant:
            if self.backend == "mlx":
                self.quant_bits = want_quant
            else:
                import torch
                if torch.cuda.is_available():
                    self.quant_bits = want_quant
                else:
                    MessageProtocol.warning(
                        f"{want_quant}-bit-Quantisierung (QLoRA) geht mit PyTorch nur auf NVIDIA-GPUs "
                        "(bitsandbytes). Es wird mit normalem LoRA weitertrainiert."
                        + (" Auf dem Mac: Einstellungen → Python-Pakete → „LLM Fine-Tuning“ (installiert mlx-lm), dann Backend 'auto' oder 'mlx'."
                           if _is_apple_silicon() else ""))
        MessageProtocol.status(
            "init",
            f"Architektur: {self.model_type or 'unbekannt'} | Backend: {self.backend.upper()}"
            + (f" | QLoRA {self.quant_bits}-bit" if self.quant_bits else "")
            + (" | LoRA" if self.config.use_lora or self.quant_bits else " | volles Fine-Tuning"))

        # Auch fuer MLX: die Datenaufbereitung laeuft ueber den HF-Tokenizer,
        # das MLX-Modell bringt beim Laden seinen eigenen mit.
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.template_source = D.ensure_chat_template(self.tokenizer)
        if self.template_source == "fallback":
            MessageProtocol.status(
                "init", "Modell ohne Chat-Template (Basismodell) — es wird ein einfaches "
                        "'### User / ### Assistant'-Format verwendet und mitgespeichert.")
        return True

    def _choose_backend(self, want_quant: int) -> str:
        requested = self.backend if self.backend in ("auto", "torch", "mlx") else "auto"
        mlx_problem = _mlx_supports(self.model_path)
        if requested == "mlx":
            if mlx_problem:
                raise ValueError(f"Backend 'mlx' nicht moeglich: {mlx_problem}.")
            return "mlx"
        if requested == "torch":
            return "torch"
        if mlx_problem is None:
            return "mlx"
        if _is_apple_silicon() and want_quant:
            MessageProtocol.warning(f"MLX nicht verfuegbar: {mlx_problem}.")
        return "torch"

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def load_data(self) -> bool:
        MessageProtocol.status("loading_data", f"Lade Dataset: {Path(self.config.dataset_path).name}")
        self.data = D.load_examples(self.config.dataset_path, self.pc, seed=self.config.seed)
        for note in self.data.notes:
            MessageProtocol.status("loading_data", note)
        kind = {"chat": "Chat/Frage-Antwort (Loss nur auf den Antworten)",
                "preference": "Praeferenzpaare chosen/rejected (DPO)",
                "text": "reiner Text (weiteres Vortraining)"}[self.data.kind]
        if self.data.kind == "preference":
            return self._load_preferences(kind)
        MessageProtocol.status(
            "loading_data",
            f"Format: {self.data.source_format} → {kind} | Train: {len(self.data.train)} | "
            f"Val: {len(self.data.val)}")

        max_len = int(self.config.max_seq_length or 512)
        train_tok, st = D.tokenize_all(self.tokenizer, self.data.train, max_len)
        val_tok, _ = D.tokenize_all(self.tokenizer, self.data.val, max_len)
        if st["dropped"]:
            MessageProtocol.warning(
                f"{st['dropped']} Beispiele verworfen: die Antwort liegt komplett hinter "
                f"max_seq_length={max_len}. Laengere Sequenzen erlauben oder Prompts kuerzen.")
        if st["truncated"]:
            MessageProtocol.status("loading_data", f"{st['truncated']} Beispiele auf {max_len} Tokens gekappt.")
        if st["unmasked"]:
            MessageProtocol.warning(
                f"Bei {st['unmasked']} Beispielen liess sich die Antwort im Chat-Template nicht "
                "abgrenzen — dort wird der ganze Text trainiert.")
        if not train_tok:
            raise ValueError("Nach der Tokenisierung ist kein Trainingsbeispiel uebrig.")
        self.train_dataset, self.eval_dataset = train_tok, val_tok
        n_tokens = sum(sum(1 for l in t.labels if l != -100) for t in train_tok)
        MessageProtocol.status("loading_data", f"Tokenisiert: {len(train_tok)} Sequenzen, {n_tokens} Ziel-Tokens")
        return True

    def _load_preferences(self, kind: str) -> bool:
        if self.backend == "mlx":
            # mlx-lm kennt nur SFT. DPO braucht das Referenzmodell (Basis ohne
            # Adapter) — das gibt es sauber nur im peft-Weg.
            MessageProtocol.warning("DPO laeuft ueber PyTorch (mlx-lm kann kein DPO) — Backend gewechselt.")
            self.backend = "torch"
            import torch
            if self.quant_bits and not torch.cuda.is_available():
                MessageProtocol.warning("4-bit gibt es mit PyTorch nur auf NVIDIA — DPO laeuft mit normalem LoRA.")
                self.quant_bits = 0
        MessageProtocol.status(
            "loading_data",
            f"Format: {self.data.source_format} → {kind} | Train: {len(self.data.train)} | Val: {len(self.data.val)}")
        max_len = int(self.config.max_seq_length or 512)
        train_p, st = D.tokenize_preferences(self.tokenizer, self.data.train, max_len)
        val_p, _ = D.tokenize_preferences(self.tokenizer, self.data.val, max_len)
        if st["dropped"]:
            MessageProtocol.warning(f"{st['dropped']} Paare verworfen (Antwort hinter max_seq_length={max_len}).")
        if not train_p:
            raise ValueError("Nach der Tokenisierung ist kein Praeferenzpaar uebrig.")
        self.train_dataset, self.eval_dataset = train_p, val_p
        self.dpo_beta = float(self.pc.get("dpo_beta", 0.1) or 0.1)
        self.dpo_sft_weight = float(self.pc.get("dpo_sft_weight", 1.0))
        MessageProtocol.status(
            "loading_data",
            f"Tokenisiert: {len(train_p)} Paare | DPO beta={self.dpo_beta}, SFT-Anteil={self.dpo_sft_weight}")
        return True

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> bool:
        if self.backend == "mlx":
            return self._build_mlx()
        return self._build_torch()

    def _torch_dtype(self):
        import torch
        pref = str(self.pc.get("precision", "auto")).lower()
        if pref == "fp32":
            return torch.float32
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Kleine Modelle in fp32 (stabiler, Speicher egal), grosse in bf16.
            n = self._approx_params()
            return torch.bfloat16 if (pref == "bf16" or n > 1.5e9) else torch.float32
        return torch.float32

    def _approx_params(self) -> float:
        c = self.model_cfg
        h, l, v = c.get("hidden_size", 0), c.get("num_hidden_layers", 0), c.get("vocab_size", 0)
        inter = c.get("intermediate_size", 4 * h)
        return float(l * (4 * h * h + 3 * h * inter) + v * h) if h and l else 0.0

    def _build_torch(self) -> bool:
        import torch
        from transformers import AutoModelForCausalLM

        self.device_used = hft.device_name()
        kwargs: Dict[str, Any] = {"dtype": self._torch_dtype()}
        if self.quant_bits:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
            except ImportError:
                raise D_deps.missing("bitsandbytes", what="QLoRA (4/8 bit) auf NVIDIA")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.quant_bits == 4, load_in_8bit=self.quant_bits == 8,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=kwargs["dtype"],
                bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = {"": 0}
        MessageProtocol.status("building_model", f"Lade Modell ({kwargs['dtype']}) ...")
        model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **kwargs)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = self.tokenizer.pad_token_id
        self.total_params = sum(p.numel() for p in model.parameters())

        if self.config.use_lora or self.quant_bits:
            try:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            except ImportError:
                raise D_deps.missing("peft", what="LoRA")
            if self.quant_bits:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=self.config.gradient_checkpointing)
            targets = list(self.config.lora_target_modules or []) or "all-linear"
            lcfg = LoraConfig(
                r=int(self.config.lora_r), lora_alpha=int(self.config.lora_alpha),
                lora_dropout=float(self.config.lora_dropout), target_modules=targets,
                bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, lcfg)
            if self.config.gradient_checkpointing:
                model.enable_input_require_grads()
        elif self.total_params > 1.5e9:
            MessageProtocol.warning(
                f"Volles Fine-Tuning von {self.total_params/1e9:.1f} Mrd. Parametern braucht grob "
                f"{self.total_params*16/1e9:.0f} GB Speicher. LoRA einschalten spart fast alles davon.")
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not self.quant_bits:
            model.to(self.device_used)
        self.model = model
        MessageProtocol.status(
            "building_model",
            f"Modell geladen | {self.total_params/1e6:.0f}M Parameter | trainierbar: "
            f"{self.trainable_params/1e6:.2f}M ({100*self.trainable_params/max(self.total_params,1):.2f} %)"
            f" | Geraet: {self.device_used.upper()}")
        return True

    def _build_mlx(self) -> bool:
        from mlx_lm.utils import load, quantize_model

        MessageProtocol.status("building_model", "Lade Modell in MLX ...")
        model, tokenizer, cfg = load(str(self.model_path), return_config=True)
        self._mlx_cfg = cfg
        if self.template_source == "fallback":
            tokenizer._tokenizer.chat_template = D.FALLBACK_CHAT_TEMPLATE
        if "quantization" in cfg and not self.quant_bits:
            # Vorquantisiertes Modell: beim Export zurueck in 16 bit, sonst kann
            # transformers (Test, Labor) die Gewichte nicht lesen.
            self.quant_bits = int(cfg["quantization"].get("bits", 4))
        if self.quant_bits and "quantization" not in cfg:
            MessageProtocol.status("building_model", f"Quantisiere auf {self.quant_bits} bit (QLoRA) ...")
            model, cfg = quantize_model(model, cfg, group_size=64, bits=self.quant_bits)
            self._mlx_cfg = cfg
        self.model, self._mlx_tok = model, tokenizer
        self.device_used = "mlx"
        return True

    # ── Generierung (Auswertung vorher/nachher) ─────────────────────────────
    def _eval_examples(self) -> List[D.Example]:
        if not self.eval_samples or not self.data or self.data.kind not in ("chat", "preference"):
            return []
        pool = self.data.test or self.data.val
        return pool[: self.eval_samples]

    def _generate_torch(self, prompts: List[List[Dict[str, str]]]) -> List[str]:
        import torch
        model = self.model
        was_training = model.training
        model.eval()
        outs = []
        use_cache = getattr(model.config, "use_cache", True)
        model.config.use_cache = True
        for msgs in prompts:
            text = D.render_chat(self.tokenizer, msgs, add_generation_prompt=True)
            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                     pad_token_id=self.tokenizer.pad_token_id)
            outs.append(self.tokenizer.decode(gen[0][enc["input_ids"].shape[1]:],
                                              skip_special_tokens=True).strip())
        model.config.use_cache = use_cache
        if was_training:
            model.train()
        return outs

    def _generate_mlx(self, prompts: List[List[Dict[str, str]]]) -> List[str]:
        from mlx_lm import generate
        self.model.eval()
        outs = []
        for msgs in prompts:
            text = D.render_chat(self._mlx_tok, msgs, add_generation_prompt=True)
            out = generate(self.model, self._mlx_tok, prompt=text, max_tokens=self.max_new_tokens, verbose=False)
            outs.append(out.strip())
        return outs

    def _run_generation_eval(self, label: str) -> Dict[str, float]:
        exs = self._eval_examples()
        if not exs:
            return {}
        MessageProtocol.status("evaluating", f"{label}: erzeuge {len(exs)} Antworten zum Vergleich ...")
        prompts = [e.prompt_messages() for e in exs]
        refs = [e.reference() or "" for e in exs]
        gen = self._generate_mlx(prompts) if self.backend == "mlx" else self._generate_torch(prompts)
        scores = D.score_generations(gen, refs)
        if label.startswith("Nach"):
            self.samples = [{"prompt": p[-1]["content"] if p else "", "reference": r, "generated": g}
                            for p, r, g in zip(prompts, refs, gen)]
        return scores

    # ── 4. Training ─────────────────────────────────────────────────────────
    def train(self) -> bool:
        if self.eval_baseline:
            self.baseline = self._run_generation_eval("Vor dem Training")
            if self.baseline:
                MessageProtocol.status(
                    "evaluating",
                    f"Vorher: Exact Match {self.baseline['exact_match']:.0%}, ROUGE-L {self.baseline['rougeL']:.2f}")
        if self.is_stopped:
            return False
        self._start_time = time.time()
        ok = self._train_mlx() if self.backend == "mlx" else self._train_torch()
        if ok:
            MessageProtocol.status("training", "Training abgeschlossen")
        return ok

    def _steps_per_epoch(self) -> int:
        eff = max(1, int(self.config.batch_size)) * max(1, int(self.config.gradient_accumulation_steps))
        return max(1, math.ceil(len(self.train_dataset) / eff))

    def _train_torch(self) -> bool:
        if self.data and self.data.kind == "preference":
            return self._train_dpo()
        import torch
        from datasets import Dataset
        from transformers import Trainer, TrainerCallback, TrainingArguments

        def to_ds(items):
            return Dataset.from_list([{"input_ids": t.input_ids, "labels": t.labels} for t in items])

        pad_id = self.tokenizer.pad_token_id

        def collate(batch):
            n = max(len(b["input_ids"]) for b in batch)
            ids = [b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch]
            lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
            att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
            return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                    "attention_mask": torch.tensor(att)}

        train_ds = to_ds(self.train_dataset)
        eval_ds = to_ds(self.eval_dataset) if self.eval_dataset else None
        eval_ds = hft.cap_eval_dataset(eval_ds, getattr(self.config, "max_eval_samples", 0), self.config.seed) \
            if eval_ds is not None else None
        total = self._steps_per_epoch() * max(1, int(self.config.epochs))
        if int(self.config.max_steps) > 0:
            total = int(self.config.max_steps)

        overrides: Dict[str, Any] = {"remove_unused_columns": False, "label_smoothing_factor": 0.0}
        if eval_ds is None:
            overrides["eval_strategy"] = "no"
        args = hft.build_training_arguments(
            self.config, self.config.effective_output_dir(), TrainingArguments, **overrides)
        self._trainer = Trainer(
            model=self.model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=collate,
            callbacks=[hft.progress_callback(TrainerCallback, self, total)])
        self._trainer.train()
        self._steps_done = int(self._trainer.state.global_step)
        return not self.is_stopped

    # ── DPO ────────────────────────────────────────────────────────────────
    @staticmethod
    def _seq_logps(model, ids, att, labels, with_counts: bool = False):
        """Summe der Log-Wahrscheinlichkeiten der Antwort-Tokens je Sequenz."""
        import torch
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        tgt = labels[:, 1:]
        mask = tgt != -100
        lp = torch.gather(logits.log_softmax(-1), 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        sums = (lp * mask).sum(-1)
        return (sums, mask.sum(-1).clamp(min=1)) if with_counts else sums

    def _ref_logps(self, model, ids, att, labels):
        import torch
        with torch.no_grad():
            if hasattr(model, "disable_adapter"):
                # LoRA: das Referenzmodell ist die Basis ohne Adapter — kein
                # zweites Modell im Speicher.
                with model.disable_adapter():
                    return self._seq_logps(model, ids, att, labels)
            return self._seq_logps(self._ref_model, ids, att, labels)

    def _dpo_collate(self, batch):
        import torch
        pad_id = self.tokenizer.pad_token_id
        seqs = [(b["c_ids"], b["c_lab"]) for b in batch] + [(b["r_ids"], b["r_lab"]) for b in batch]
        n = max(len(i) for i, _ in seqs)
        return {
            "input_ids": torch.tensor([i + [pad_id] * (n - len(i)) for i, _ in seqs]),
            "labels": torch.tensor([l + [-100] * (n - len(l)) for _, l in seqs]),
            "attention_mask": torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in seqs]),
        }

    def _dpo_batch(self, model, inputs):
        """(loss, Anteil richtig geordneter Paare, mittlerer Abstand)."""
        import torch.nn.functional as F
        ids, att, lab = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        half = ids.shape[0] // 2
        pol, counts = self._seq_logps(model, ids, att, lab, with_counts=True)
        ref = self._ref_logps(model, ids, att, lab)
        margin = self.dpo_beta * ((pol[:half] - ref[:half]) - (pol[half:] - ref[half:]))
        loss = -F.logsigmoid(margin).mean()
        if self.dpo_sft_weight > 0:
            # RPO: reines DPO senkt auch Tokens, die gute und schlechte Antwort
            # teilen ("Abteilung: … | Code: FT-") — im echten Lauf zerfiel so das
            # Format (Exact Match 30 % -> 0 %). Der SFT-Anteil auf der guten
            # Antwort haelt es stabil.
            loss = loss + self.dpo_sft_weight * (-(pol[:half] / counts[:half]).mean())
        return loss, float((margin > 0).float().mean()), float(margin.mean())

    def _train_dpo(self) -> bool:
        import copy
        from datasets import Dataset
        from transformers import Trainer, TrainerCallback, TrainingArguments

        plugin = self
        if not hasattr(self.model, "disable_adapter"):
            MessageProtocol.warning("DPO ohne LoRA haelt eine zweite Kopie des Modells als Referenz im Speicher.")
            self._ref_model = copy.deepcopy(self.model).eval()
            for p in self._ref_model.parameters():
                p.requires_grad_(False)

        def to_ds(pairs):
            return Dataset.from_list([{"c_ids": p.chosen.input_ids, "c_lab": p.chosen.labels,
                                       "r_ids": p.rejected.input_ids, "r_lab": p.rejected.labels} for p in pairs])

        class _DPOTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                loss, acc, margin = plugin._dpo_batch(model, inputs)
                plugin._last_pref_acc, plugin._last_margin = acc, margin
                return (loss, {}) if return_outputs else loss

        train_ds = to_ds(self.train_dataset)
        eval_ds = to_ds(self.eval_dataset) if self.eval_dataset else None
        total = self._steps_per_epoch() * max(1, int(self.config.epochs))
        if int(self.config.max_steps) > 0:
            total = int(self.config.max_steps)
        overrides: Dict[str, Any] = {"remove_unused_columns": False, "label_smoothing_factor": 0.0,
                                     "prediction_loss_only": True}
        if eval_ds is None:
            overrides["eval_strategy"] = "no"
        args = hft.build_training_arguments(self.config, self.config.effective_output_dir(), TrainingArguments, **overrides)
        self._trainer = _DPOTrainer(
            model=self.model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=self._dpo_collate, callbacks=[hft.progress_callback(TrainerCallback, self, total)])
        self._trainer.train()
        self._steps_done = int(self._trainer.state.global_step)
        return not self.is_stopped

    def _dpo_eval(self) -> Dict[str, float]:
        """Anteil der Val-Paare, bei denen das Modell chosen staerker bevorzugt als die Basis."""
        import torch
        pairs = self.eval_dataset or []
        if not pairs:
            return {}
        self.model.eval()
        accs, margins = [], []
        for i in range(0, len(pairs), 4):
            chunk = pairs[i:i + 4]
            batch = self._dpo_collate([{"c_ids": p.chosen.input_ids, "c_lab": p.chosen.labels,
                                        "r_ids": p.rejected.input_ids, "r_lab": p.rejected.labels} for p in chunk])
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                _, acc, margin = self._dpo_batch(self.model, batch)
            accs.append(acc * len(chunk))
            margins.append(margin * len(chunk))
        return {"preference_accuracy": sum(accs) / len(pairs), "reward_margin": sum(margins) / len(pairs)}

    def _train_mlx(self) -> bool:
        import mlx.optimizers as optim
        from mlx_lm.tuner.callbacks import TrainingCallback
        from mlx_lm.tuner.trainer import TrainingArgs, train as mlx_train
        from mlx_lm.tuner.utils import linear_to_lora_layers

        plugin = self
        model = self.model
        model.freeze()
        n_layers = len(model.layers)
        want = int(self.pc.get("lora_layers", -1) or -1)
        num_layers = n_layers if want <= 0 else min(want, n_layers)
        if self.config.use_lora or self.quant_bits:
            r = int(self.config.lora_r)
            linear_to_lora_layers(model, num_layers, {
                "rank": r, "dropout": float(self.config.lora_dropout),
                # peft skaliert mit alpha/r — gleiche Lernraten fuer beide Backends
                "scale": float(self.config.lora_alpha) / max(r, 1),
            })
        else:
            for layer in model.layers[-num_layers:]:
                layer.unfreeze()
        from mlx.utils import tree_flatten
        self.trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        self.total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
        MessageProtocol.status(
            "building_model",
            f"LoRA in {num_layers}/{n_layers} Bloecken | trainierbar: {self.trainable_params/1e6:.2f}M")

        bs = max(1, min(int(self.config.batch_size), len(self.train_dataset)))
        train_set = _mlx_dataset(self.train_dataset, bs)
        val_set = _mlx_dataset(self.eval_dataset or self.train_dataset, bs)
        spe = max(1, math.ceil(len(self.train_dataset) / bs))
        iters = int(self.config.max_steps) if int(self.config.max_steps) > 0 else spe * max(1, int(self.config.epochs))
        report_every = max(1, min(int(self.config.logging_steps or 10), max(1, iters // 20)))
        eval_every = max(1, spe if str(self.config.eval_strategy).lower() == "epoch" else int(self.config.eval_steps or spe))
        adapter_dir = Path(self.config.effective_output_dir()) / "mlx_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        lr = float(self.config.learning_rate)
        sched = str(self.config.scheduler or "linear").lower()
        if sched in ("cosine", "linear") and iters > 1:
            import mlx.optimizers as _o
            warm = int(self.config.warmup_steps or 0) or int(round(iters * float(self.config.warmup_ratio or 0)))
            decay = (_o.cosine_decay(lr, max(iters - warm, 1)) if sched == "cosine"
                     else _o.linear_schedule(lr, 0.0, max(iters - warm, 1)))
            lr_sched = _o.join_schedules([_o.linear_schedule(0.0, lr, warm), decay], [warm]) if warm else decay
        else:
            lr_sched = lr
        opt_name = str(self.config.optimizer or "adamw").lower()
        if opt_name == "sgd":
            opt = optim.SGD(learning_rate=lr_sched)
        elif opt_name == "adafactor":
            opt = optim.Adafactor(learning_rate=lr_sched)
        else:
            opt = optim.AdamW(learning_rate=lr_sched, weight_decay=float(self.config.weight_decay))

        epochs = max(1, int(self.config.epochs))

        class _CB(TrainingCallback):
            def on_train_loss_report(self, info):
                it = int(info.get("iteration", 0))
                plugin._last_train_loss = float(info.get("train_loss", 0.0))
                plugin._last_lr = float(info.get("learning_rate", lr))
                plugin._steps_done = it
                MessageProtocol.progress(
                    epoch=min(epochs, it // spe + 1), total_epochs=epochs, step=it, total_steps=iters,
                    train_loss=plugin._last_train_loss, val_loss=plugin._last_val_loss,
                    learning_rate=plugin._last_lr,
                    metrics={"tokens_per_second": float(info.get("tokens_per_second", 0.0)),
                             "peak_memory_gb": float(info.get("peak_memory", 0.0))})
                if plugin.is_stopped:
                    raise _StopTraining()

            def on_val_loss_report(self, info):
                # mlx-lm misst einmal vor dem ersten Schritt (Iteration 0). Als
                # aktueller Wert stand er sonst bis zur ersten echten Messung
                # in jedem Fortschritt — die Analyse zeigte ihn als Val-Loss
                # von Epoche 1 (6.27 statt 0.58). Er ist der Vorher-Wert.
                loss = float(info.get("val_loss", 0.0))
                if int(info.get("iteration", 0)) <= 0:
                    plugin._baseline_val_loss = loss
                else:
                    plugin._last_val_loss = loss
                    MessageProtocol.status("training", f"Val-Loss {loss:.4f} (Schritt {info.get('iteration')})")
                if plugin.is_stopped:
                    raise _StopTraining()

        args = TrainingArgs(
            batch_size=bs, iters=iters, val_batches=-1 if len(self.eval_dataset or []) <= 200 else 25,
            steps_per_report=report_every, steps_per_eval=eval_every,
            steps_per_save=max(int(self.config.save_steps or 100), 50),
            adapter_file=str(adapter_dir / "adapters.safetensors"),
            max_seq_length=int(self.config.max_seq_length or 512),
            grad_checkpoint=bool(self.config.gradient_checkpointing),
            grad_accumulation_steps=max(1, int(self.config.gradient_accumulation_steps)))
        MessageProtocol.status("training", f"MLX: {iters} Schritte, Batch {bs}, {spe} Schritte je Epoche")
        try:
            mlx_train(model=model, optimizer=opt, train_dataset=train_set, val_dataset=val_set,
                      args=args, training_callback=_CB())
        except _StopTraining:
            return False
        return True

    # ── 5. Validierung ──────────────────────────────────────────────────────
    def validate(self) -> Dict[str, Any]:
        if self.backend == "mlx":
            val_loss = self._mlx_val_loss()
            result = {"eval_loss": val_loss}
        else:
            result = self._trainer.evaluate() if self._trainer and self._trainer.eval_dataset is not None else {}
        self.after = self._run_generation_eval("Nach dem Training")
        if self.data and self.data.kind == "preference":
            dpo = self._dpo_eval()
            self.after.update(dpo)
            if dpo:
                MessageProtocol.status(
                    "validating",
                    f"DPO: bevorzugt bei {dpo['preference_accuracy']:.0%} der Val-Paare die bessere Antwort "
                    f"(mittlerer Abstand {dpo['reward_margin']:.3f})")

        class _S:  # final_metrics erwartet trainer.state
            pass
        fake = _S()
        fake.state = _S()
        fake.state.epoch = self.config.epochs
        fake.state.global_step = self._steps_done
        trainer = self._trainer if self._trainer is not None else fake
        metrics = hft.final_metrics(self, trainer, result, self._start_time,
                                    architecture=self.model_type, num_labels=0)
        for key in ("accuracy", "f1", "precision", "recall", "num_labels"):
            metrics.pop(key, None)
        vl = result.get("eval_loss")
        if vl:
            metrics["final_val_loss"] = float(vl)
            metrics["perplexity"] = float(math.exp(min(float(vl), 50)))
        if self.after:
            metrics.update({k: float(v) for k, v in self.after.items()})
        if self.baseline:
            metrics.update({f"baseline_{k}": float(v) for k, v in self.baseline.items()})
        if self._baseline_val_loss is not None and "baseline_perplexity" not in metrics:
            # Die Analyse zeigt die Perplexitaet dann mit Vorher-Wert.
            metrics["baseline_perplexity"] = float(math.exp(min(float(self._baseline_val_loss), 50)))
        metrics.update({
            "backend": self.backend, "device": self.device_used,
            "trainable_params": int(self.trainable_params), "total_params": int(self.total_params),
            "lora": bool(self.config.use_lora or self.quant_bits), "quant_bits": int(self.quant_bits),
            "chat_template": self.template_source, "data_kind": self.data.kind if self.data else "",
        })
        if self.after.get("exact_match") is not None:
            msg = f"Nachher: Exact Match {self.after['exact_match']:.0%}, ROUGE-L {self.after['rougeL']:.2f}"
            if self.baseline:
                msg += f" (vorher {self.baseline['exact_match']:.0%} / {self.baseline['rougeL']:.2f})"
            MessageProtocol.status("validating", msg)
        return metrics

    def _mlx_val_loss(self) -> float:
        from mlx_lm.tuner.trainer import evaluate

        items = self.eval_dataset or self.train_dataset
        bs = max(1, min(int(self.config.batch_size), len(items)))
        return float(evaluate(model=self.model, dataset=_mlx_dataset(items, bs), batch_size=bs,
                              num_batches=-1, max_seq_length=int(self.config.max_seq_length or 512)))

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        if self.backend == "mlx":
            self._export_mlx(out)
        else:
            self._export_torch(out)
        meta = {
            "task": "causal_lm", "base_model": str(self.model_path), "model_type": self.model_type,
            "backend": self.backend, "lora": bool(self.config.use_lora or self.quant_bits),
            "lora_r": int(self.config.lora_r), "lora_alpha": int(self.config.lora_alpha),
            "quant_bits": int(self.quant_bits), "chat_template": self.template_source,
            "system_prompt": str(self.pc.get("system_prompt") or ""),
            "max_seq_length": int(self.config.max_seq_length or 512),
            "data_kind": self.data.kind if self.data else "",
        }
        (out / "frametrain_llm.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.samples:
            with open(out / "samples.jsonl", "w", encoding="utf-8") as fh:
                for s in self.samples:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        if self.pc.get("export_gguf"):
            self._export_gguf(out)
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)

    def _export_torch(self, out: Path) -> None:
        import torch
        model = self.model
        if hasattr(model, "peft_config"):
            model.save_pretrained(str(out / "adapter"))
            MessageProtocol.status("export", "LoRA-Adapter gespeichert, fuehre ihn mit dem Basismodell zusammen ...")
            if self.quant_bits:
                # Ein 4-bit-Modell laesst sich nicht verlustfrei zusammenfuehren:
                # Basismodell in 16 bit neu laden und den Adapter dort einrechnen.
                from peft import PeftModel
                from transformers import AutoModelForCausalLM
                base = AutoModelForCausalLM.from_pretrained(str(self.model_path), dtype=torch.bfloat16)
                model = PeftModel.from_pretrained(base, str(out / "adapter"))
            model = model.merge_and_unload()
        orig = self.model_cfg.get("torch_dtype") or self.model_cfg.get("dtype")
        target = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(str(orig))
        if target is not None:
            model = model.to(target)
        model.config.use_cache = True
        model.save_pretrained(str(out), safe_serialization=True)
        self.tokenizer.save_pretrained(str(out))
        gen_cfg = self.model_path / "generation_config.json"
        if gen_cfg.exists() and not (out / "generation_config.json").exists():
            (out / "generation_config.json").write_text(gen_cfg.read_text(encoding="utf-8"), encoding="utf-8")

    def _export_mlx(self, out: Path) -> None:
        import shutil
        from mlx.utils import tree_unflatten
        from mlx_lm.utils import dequantize_model, save

        model = self.model
        adapter_dir = Path(self.config.effective_output_dir()) / "mlx_adapter"
        if adapter_dir.exists():
            shutil.copytree(adapter_dir, out / "adapter", dirs_exist_ok=True)
        fused = [(n, m.fuse(dequantize=bool(self.quant_bits))) for n, m in model.named_modules() if hasattr(m, "fuse")]
        if fused:
            model.update_modules(tree_unflatten(fused))
        cfg = dict(self._mlx_cfg)
        if self.quant_bits:
            # Fuer Test, Labor und transformers ohne MLX: zurueck in 16 bit.
            model = dequantize_model(model)
            cfg.pop("quantization", None)
            cfg.pop("quantization_config", None)
        save(out, str(self.model_path), model, self._mlx_tok, cfg, donate_model=False)
        if self.template_source == "fallback":
            # save() schreibt den MLX-Tokenizer; das Ersatz-Template muss mit.
            self.tokenizer.save_pretrained(str(out))

    def _export_gguf(self, out: Path) -> None:
        """GGUF fuer Ollama / LM Studio (ft_data.gguf_export) plus Ollama-Modelfile."""
        from ft_data import gguf_export
        outtype = str(self.pc.get("gguf_type", "q8_0") or "q8_0")
        target = out / "model.gguf"
        ok, msg = gguf_export.convert(out, target, outtype,
                                      status=lambda m: MessageProtocol.status("export", m))
        if not ok:
            MessageProtocol.warning(msg)
            return
        gguf_export.write_modelfile(out, target.name, str(self.pc.get("system_prompt") or ""))
        MessageProtocol.status("export", msg + " — für Ollama: ollama create mein-modell -f Modelfile")

    def get_metrics(self) -> Dict[str, Any]:
        return {"architecture": self.model_type, "device": self.device_used}
