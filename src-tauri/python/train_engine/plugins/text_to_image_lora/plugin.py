"""
Text-to-Image LoRA (Stable Diffusion 1.x/2.x, SDXL)
===================================================
Bringt einem Diffusionsmodell einen eigenen Stil oder ein eigenes Objekt bei.
Trainiert wird nur ein LoRA-Adapter im UNet (to_q/to_k/to_v/to_out.0) — ein
volles Fine-Tuning braucht fuer SD 1.5 mehr als 20 GB und ist lokal nicht
drin. Der Adapter ist wenige MB gross.

Ablauf je Schritt (wie diffusers train_text_to_image_lora.py):
    Bild -> VAE -> Latents * scaling_factor
    Rauschen + zufaelliger Timestep (DDPMScheduler des Modells)
    UNet(+LoRA) sagt Rauschen (epsilon) bzw. v (v_prediction, SD 2.x) voraus
    Loss = MSE gegen das Ziel

Datenformate (siehe ft_data.image_text):
    Bilder + gleichnamige .txt-Captions | metadata.jsonl/.csv (file_name + text)
    | nur Bilder + plugin_config instance_prompt ("a photo of sks dog", DreamBooth)

Die Schleife ist bewusst selbst geschrieben statt ueber den HF-Trainer: der
kennt weder VAE noch Scheduler, und fuer die paar Bausteine lohnt kein
accelerate-Unterbau.
"""
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import TrainingConfig
from core.plugin_base import TrainPlugin
from core.protocol import MessageProtocol

from ft_data.diffusion import (
    LORA_INFO_FILE, LORA_WEIGHTS_FILE, generate_image, pipeline_kind, read_model_index,
    unsupported_message,
)
from ft_data.image_text import ImageTextSample, resolve_image_text

# Attention-Projektionen im UNet — dieselbe Auswahl wie im diffusers-Referenzskript.
LORA_TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]


def training_steps(n_samples: int, batch_size: int, grad_accum: int, epochs: int, max_steps: int) -> int:
    """Optimierer-Schritte insgesamt. max_steps > 0 hat Vorrang vor den Epochen."""
    if max_steps and max_steps > 0:
        return int(max_steps)
    per_epoch = math.ceil(max(n_samples, 1) / max(batch_size, 1))
    return max(1, math.ceil(per_epoch / max(grad_accum, 1)) * max(epochs, 1))


def crop_box(width: int, height: int, resolution: int, center: bool, rng: random.Random):
    """Skaliert die kuerzere Seite auf resolution und waehlt den Ausschnitt.

    Rueckgabe: (neue Breite, neue Hoehe, links, oben). SDXL bekommt die
    Ausschnitt-Koordinaten als Konditionierung mit.
    """
    scale = resolution / min(width, height)
    new_w = max(resolution, round(width * scale))
    new_h = max(resolution, round(height * scale))
    if center:
        left = (new_w - resolution) // 2
        top = (new_h - resolution) // 2
    else:
        left = rng.randint(0, new_w - resolution)
        top = rng.randint(0, new_h - resolution)
    return new_w, new_h, left, top


class Plugin(TrainPlugin):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        pc = config.plugin_config or {}
        self.resolution = int(pc.get("resolution", 512) or 512)
        self.instance_prompt = str(pc.get("instance_prompt", "") or "").strip()
        self.validation_prompt = str(pc.get("validation_prompt", "") or "").strip()
        self.num_validation_images = max(0, int(pc.get("num_validation_images", 2) or 0))
        self.sample_steps = max(1, int(pc.get("sample_steps", 25) or 25))
        self.guidance_scale = float(pc.get("guidance_scale", 7.5) or 7.5)
        self.center_crop = bool(pc.get("center_crop", True))
        self.random_flip = bool(pc.get("random_flip", True))
        self.merge_lora = bool(pc.get("merge_lora", False))
        self.sample_before = bool(pc.get("sample_before_training", True))
        self.noise_offset = float(pc.get("noise_offset", 0.0) or 0.0)
        self.checkpoint_every = int(pc.get("checkpoint_every", 0) or 0) or max(int(config.save_steps or 0), 0)

        self.pipe = None
        self.kind = "sd"
        self.pipeline_class = ""
        self.noise_scheduler = None
        self.device = None
        self.device_used = "cpu"
        self.weight_dtype = None
        self.train_dataset: List[ImageTextSample] = []
        self.eval_dataset: List[ImageTextSample] = []
        self.losses: List[float] = []
        self.global_step = 0
        self.total_steps = 0
        self.epochs_done = 0
        self.samples_written: List[str] = []
        self.fixed_loss_before: Optional[float] = None
        self._embed_cache: Dict[str, Any] = {}
        self._start_time = time.time()

    # ── 1. Setup ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        try:
            import diffusers  # noqa: F401
            import peft  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"{exc}. Fuer Diffusion-LoRA fehlen Pakete. Installiere: pip install diffusers peft safetensors"
            )
        model_path = Path(self.config.model_path)
        index = read_model_index(model_path)
        if not index:
            MessageProtocol.error(
                "Keine diffusers-Pipeline",
                f"In {model_path} liegt keine model_index.json. Erwartet wird ein Stable-Diffusion-"
                "Modell im diffusers-Format (model_index.json + Ordner unet/, vae/, text_encoder/, "
                "tokenizer/, scheduler/), z. B. 'stable-diffusion-v1-5/stable-diffusion-v1-5' oder "
                "'segmind/tiny-sd'. Einzelne .ckpt/.safetensors-Checkpoints werden nicht gelesen.",
            )
            return False
        self.pipeline_class = index.get("_class_name", "")
        kind = pipeline_kind(self.pipeline_class)
        if kind is None:
            MessageProtocol.error("Pipeline nicht unterstuetzt", unsupported_message(self.pipeline_class))
            return False
        self.kind = kind
        if self.resolution % 8:
            self.resolution = max(64, self.resolution - self.resolution % 8)
            MessageProtocol.warning(f"Aufloesung muss durch 8 teilbar sein — nutze {self.resolution}.")
        MessageProtocol.status(
            "init",
            f"Pipeline: {self.pipeline_class} ({'SDXL' if kind == 'sdxl' else 'SD 1.x/2.x'}) | "
            f"Aufloesung {self.resolution}px | LoRA r={self.config.lora_r}, alpha={self.config.lora_alpha}",
        )

    # ── 2. Daten ────────────────────────────────────────────────────────────
    def load_data(self) -> None:
        splits = resolve_image_text(
            Path(self.config.dataset_path), "caption", instance_prompt=self.instance_prompt,
            seed=self.config.seed, val_fraction=0.0,
            status=lambda m: MessageProtocol.status("loading_data", m),
        )
        for note in splits.notes:
            MessageProtocol.status("loading_data", note)
        self.train_dataset = splits.train
        self.eval_dataset = splits.val
        if not self.validation_prompt:
            self.validation_prompt = self.instance_prompt or self.train_dataset[0].answer
        unique = len({s.answer for s in self.train_dataset})
        MessageProtocol.status(
            "loading_data",
            f"✓ {len(self.train_dataset)} Bilder, {unique} verschiedene Beschreibungen"
            + (f" (DreamBooth: '{self.instance_prompt}')" if self.instance_prompt and unique == 1 else ""),
        )
        if len(self.train_dataset) < 3:
            MessageProtocol.warning(
                f"Nur {len(self.train_dataset)} Bilder — fuer einen Stil oder ein Objekt sind 5–30 Bilder ueblich.")

    # ── 3. Modell ───────────────────────────────────────────────────────────
    def build_model(self) -> None:
        import torch
        from diffusers import DDPMScheduler
        from ft_data.diffusion import load_base_pipeline, torch_device

        self.device = torch_device()
        self.device_used = self.device.type
        # Eingefrorene Gewichte in fp16 nur auf CUDA und nur auf Wunsch. Auf MPS
        # liefert der VAE in fp16 NaNs; dort und auf der CPU bleibt alles fp32.
        use_half = self.device.type == "cuda" and (self.config.fp16 or self.config.bf16)
        self.weight_dtype = (torch.bfloat16 if self.config.bf16 else torch.float16) if use_half else torch.float32

        MessageProtocol.status("building_model", "Lade Pipeline (UNet, VAE, Text-Encoder)...")
        pipe, _ = load_base_pipeline(Path(self.config.model_path), torch.float32)
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.config.model_path, subfolder="scheduler")

        for comp in ("vae", "text_encoder", "text_encoder_2", "unet"):
            m = getattr(pipe, comp, None)
            if m is not None:
                m.requires_grad_(False)
        pipe.unet.to(self.device, dtype=self.weight_dtype)
        pipe.vae.to(self.device, dtype=torch.float32)  # VAE immer fp32 (NaN-sicher)
        for comp in ("text_encoder", "text_encoder_2"):
            m = getattr(pipe, comp, None)
            if m is not None:
                m.to(self.device, dtype=self.weight_dtype)
        self.pipe = pipe

        pred = getattr(self.noise_scheduler.config, "prediction_type", "epsilon")
        params = sum(p.numel() for p in pipe.unet.parameters())
        MessageProtocol.status(
            "building_model",
            f"✓ UNet {params/1e6:.0f}M Parameter | Vorhersage: {pred} | Geraet: {self.device.type.upper()}"
            f" | Gewichte: {str(self.weight_dtype).replace('torch.', '')}",
        )

    def _add_lora(self) -> None:
        import torch
        from peft import LoraConfig

        targets = list(self.config.lora_target_modules or []) or LORA_TARGETS
        lora_config = LoraConfig(
            r=int(self.config.lora_r), lora_alpha=int(self.config.lora_alpha),
            init_lora_weights="gaussian", target_modules=targets,
            lora_dropout=float(self.config.lora_dropout or 0.0),
        )
        self.pipe.unet.add_adapter(lora_config)
        # Die trainierbaren LoRA-Gewichte immer in fp32, auch wenn das UNet fp16 ist.
        for p in self.pipe.unet.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)
        if self.config.gradient_checkpointing:
            self.pipe.unet.enable_gradient_checkpointing()
        n = sum(p.numel() for p in self.pipe.unet.parameters() if p.requires_grad)
        MessageProtocol.status("training", f"LoRA-Adapter: {n/1e6:.2f}M trainierbare Parameter ({', '.join(targets)})")

    # ── Bausteine ───────────────────────────────────────────────────────────
    def _load_pixels(self, sample: ImageTextSample, rng: random.Random):
        """Bild -> Tensor [-1, 1] in resolution x resolution, plus SDXL-Zeitangaben."""
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(sample.image) as img:
            img = img.convert("RGB")
            w, h = img.size
            new_w, new_h, left, top = crop_box(w, h, self.resolution, self.center_crop, rng)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            img = img.crop((left, top, left + self.resolution, top + self.resolution))
        if self.random_flip and rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        pixels = torch.from_numpy(arr).permute(2, 0, 1)
        # SDXL-Konditionierung: Originalgroesse, Ausschnitt oben/links, Zielgroesse.
        time_ids = [h, w, round(top * h / new_h), round(left * w / new_w), self.resolution, self.resolution]
        return pixels, time_ids

    def _encode_prompt(self, caption: str):
        """Text-Embeddings (gecacht — der Text-Encoder ist eingefroren)."""
        import torch

        hit = self._embed_cache.get(caption)
        if hit is not None:
            return hit
        with torch.no_grad():
            if self.kind == "sdxl":
                emb, _, pooled, _ = self.pipe.encode_prompt(
                    prompt=caption, device=self.device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False)
                out = (emb.to(self.weight_dtype), pooled.to(self.weight_dtype))
            else:
                emb, _ = self.pipe.encode_prompt(caption, self.device, 1, False)
                out = (emb.to(self.weight_dtype), None)
        if len(self._embed_cache) < 4096:
            self._embed_cache[caption] = out
        return out

    def _loss(self, batch: List[ImageTextSample], rng: random.Random):
        import torch
        import torch.nn.functional as F

        pix, time_ids = zip(*(self._load_pixels(s, rng) for s in batch))
        pixels = torch.stack(pix).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            latents = self.pipe.vae.encode(pixels).latent_dist.sample()
            latents = (latents * self.pipe.vae.config.scaling_factor).to(self.weight_dtype)

        noise = torch.randn_like(latents)
        if self.noise_offset:
            noise = noise + self.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device, dtype=latents.dtype)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)

        embs = [self._encode_prompt(s.answer) for s in batch]
        hidden = torch.cat([e[0] for e in embs], dim=0)
        kwargs: Dict[str, Any] = {}
        if self.kind == "sdxl":
            kwargs["added_cond_kwargs"] = {
                "text_embeds": torch.cat([e[1] for e in embs], dim=0),
                "time_ids": torch.tensor(time_ids, device=self.device, dtype=self.weight_dtype),
            }

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"prediction_type '{pred_type}' wird nicht unterstuetzt (epsilon, v_prediction).")

        with torch.autocast(self.device.type, dtype=self.weight_dtype,
                            enabled=self.device.type == "cuda" and self.weight_dtype != torch.float32):
            model_pred = self.pipe.unet(noisy, timesteps, hidden, return_dict=False, **kwargs)[0]
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    def _fixed_noise_loss(self) -> Optional[float]:
        """Loss mit festem Rauschen und festen Timesteps auf bis zu 8 Trainingsbildern.

        Der Schritt-Loss springt je nach zufaellig gezogenem Timestep um Faktor
        zehn und zeigt Lernfortschritt kaum. Mit festem Seed ist der Wert vor
        und nach dem Training direkt vergleichbar.
        """
        import torch
        import torch.nn.functional as F

        samples = self.train_dataset[:8]
        if not samples:
            return None
        unet = self.pipe.unet
        was_training = unet.training
        unet.eval()
        gen = torch.Generator(device="cpu").manual_seed(1234)
        rng = random.Random(1234)
        flip, self.random_flip = self.random_flip, False
        total, n = 0.0, 0
        try:
            with torch.no_grad():
                for s in samples:
                    pixels, time_ids = self._load_pixels(s, rng)
                    pixels = pixels.unsqueeze(0).to(self.device, dtype=torch.float32)
                    lat = self.pipe.vae.encode(pixels).latent_dist.mean
                    lat = (lat * self.pipe.vae.config.scaling_factor).to(self.weight_dtype)
                    emb, pooled = self._encode_prompt(s.answer)
                    kwargs: Dict[str, Any] = {}
                    if self.kind == "sdxl":
                        kwargs["added_cond_kwargs"] = {
                            "text_embeds": pooled,
                            "time_ids": torch.tensor([time_ids], device=self.device, dtype=self.weight_dtype)}
                    for t in (100, 300, 500, 800):
                        t = min(t, self.noise_scheduler.config.num_train_timesteps - 1)
                        noise = torch.randn(lat.shape, generator=gen).to(lat.device, dtype=lat.dtype)
                        ts = torch.tensor([t], device=lat.device).long()
                        noisy = self.noise_scheduler.add_noise(lat, noise, ts)
                        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" \
                            else self.noise_scheduler.get_velocity(lat, noise, ts)
                        pred = unet(noisy, ts, emb, return_dict=False, **kwargs)[0]
                        total += float(F.mse_loss(pred.float(), target.float()).item())
                        n += 1
        except Exception as exc:
            MessageProtocol.warning(f"Loss mit festem Rauschen nicht berechenbar: {exc}")
            return None
        finally:
            self.random_flip = flip
            if was_training:
                unet.train()
        return total / max(n, 1)

    def _write_samples(self, tag: str) -> None:
        """Beispielbilder mit festem Seed — vorher/nachher direkt vergleichbar."""
        if self.num_validation_images <= 0 or not self.validation_prompt:
            return
        import torch

        out = Path(self.config.output_path) / "samples"
        out.mkdir(parents=True, exist_ok=True)
        unet = self.pipe.unet
        was_training = unet.training
        unet.eval()
        MessageProtocol.status(
            "validating" if tag == "after" else "training",
            f"Erzeuge {self.num_validation_images} Beispielbild(er) ({'vor' if tag == 'before' else 'nach'} "
            f"dem Training): '{self.validation_prompt}'",
        )
        try:
            for i in range(self.num_validation_images):
                with torch.no_grad():
                    img = generate_image(
                        self.pipe, self.validation_prompt, steps=self.sample_steps,
                        guidance=self.guidance_scale, seed=self.config.seed + i, size=self.resolution)
                path = out / f"{tag}_{i + 1}.png"
                img.save(path)
                self.samples_written.append(str(path))
        except Exception as exc:  # Beispielbilder duerfen das Training nie kippen
            MessageProtocol.warning(f"Beispielbild konnte nicht erzeugt werden: {exc}")
        finally:
            if was_training:
                unet.train()

    def _lora_state_dict(self):
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft.utils import get_peft_model_state_dict

        return convert_state_dict_to_diffusers(get_peft_model_state_dict(self.pipe.unet))

    def _save_lora(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        type(self.pipe).save_lora_weights(
            save_directory=str(directory), unet_lora_layers=self._lora_state_dict(),
            weight_name=LORA_WEIGHTS_FILE, safe_serialization=True)
        return directory / LORA_WEIGHTS_FILE

    def _checkpoint(self, epoch: int) -> None:
        base = Path(self.config.effective_output_dir()) / "checkpoints"
        path = self._save_lora(base / f"step-{self.global_step}")
        MessageProtocol.checkpoint(self.global_step, str(path.parent), epoch=epoch,
                                   metrics={"train_loss": self.losses[-1] if self.losses else None})
        keep = max(int(self.config.save_total_limit or 0), 1)
        old = sorted((d for d in base.glob("step-*") if d.is_dir()),
                     key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else 0)
        for d in old[:-keep]:
            shutil.rmtree(d, ignore_errors=True)

    # ── 4. Training ─────────────────────────────────────────────────────────
    def train(self) -> None:
        import torch

        torch.manual_seed(self.config.seed)
        rng = random.Random(self.config.seed)
        if self.sample_before:
            self._write_samples("before")
        self.fixed_loss_before = self._fixed_noise_loss()
        self._add_lora()
        unet = self.pipe.unet
        unet.train()

        params = [p for p in unet.parameters() if p.requires_grad]
        opt_name = str(self.config.optimizer).lower()
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.config.learning_rate,
                                        momentum=self.config.sgd_momentum, weight_decay=self.config.weight_decay)
        else:
            optimizer = torch.optim.AdamW(
                params, lr=self.config.learning_rate, betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon, weight_decay=self.config.weight_decay)

        bs = max(int(self.config.batch_size), 1)
        accum = max(int(self.config.gradient_accumulation_steps), 1)
        self.total_steps = training_steps(len(self.train_dataset), bs, accum, self.config.epochs,
                                          int(self.config.max_steps))
        warmup = int(self.config.warmup_steps) or int(round(self.total_steps * float(self.config.warmup_ratio or 0)))
        from diffusers.optimization import get_scheduler
        try:
            lr_scheduler = get_scheduler(self.config.scheduler, optimizer=optimizer,
                                         num_warmup_steps=warmup, num_training_steps=self.total_steps)
        except ValueError:
            lr_scheduler = get_scheduler("constant", optimizer=optimizer)

        per_epoch = max(1, math.ceil(math.ceil(len(self.train_dataset) / bs) / accum))
        total_epochs = max(1, math.ceil(self.total_steps / per_epoch))
        MessageProtocol.status(
            "training",
            f"Training: {self.total_steps} Schritte | Batch {bs} x {accum} | ~{total_epochs} Epoche(n) | "
            f"Geraet: {self.device.type.upper()}",
        )
        self._start_time = time.time()
        epoch = 0
        done = False
        while not done:
            epoch += 1
            order = self.train_dataset[:]
            rng.shuffle(order)
            batches = [order[i:i + bs] for i in range(0, len(order), bs)]
            micro = 0
            step_losses: List[float] = []
            for batch in batches:
                if self.is_stopped:
                    done = True
                    break
                loss = self._loss(batch, rng)
                if not torch.isfinite(loss):
                    MessageProtocol.error(
                        "Loss ist NaN/unendlich",
                        "Das Training ist numerisch instabil geworden. Lernrate senken (z. B. 5e-5) "
                        "und fp16/bf16 ausschalten.")
                    return False
                (loss / accum).backward()
                step_losses.append(float(loss.detach().item()))
                micro += 1
                if micro % accum:
                    continue
                if self.config.max_grad_norm and self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                step_loss = sum(step_losses) / len(step_losses)
                step_losses = []
                self.losses.append(step_loss)
                MessageProtocol.progress(
                    epoch=min(epoch, total_epochs), total_epochs=total_epochs,
                    step=self.global_step, total_steps=self.total_steps,
                    train_loss=step_loss, learning_rate=lr_scheduler.get_last_lr()[0],
                    metrics={"loss_avg10": sum(self.losses[-10:]) / len(self.losses[-10:])},
                )
                if self.checkpoint_every and self.global_step % self.checkpoint_every == 0 \
                        and self.global_step < self.total_steps:
                    self._checkpoint(epoch)
                if self.global_step >= self.total_steps:
                    done = True
                    break
            self.epochs_done = epoch
        if self.device.type == "mps":
            torch.mps.empty_cache()
        if self.is_stopped and self.global_step > 0:
            # Abgebrochen: was bis hier gelernt wurde, liegt als Checkpoint vor.
            self._checkpoint(epoch)
            MessageProtocol.status("stopped", f"Gestoppt nach {self.global_step} Schritten — LoRA als Checkpoint gesichert.")
        else:
            MessageProtocol.status("training", "Training abgeschlossen")

    # ── 5. Validierung ──────────────────────────────────────────────────────
    def validate(self) -> Dict[str, float]:
        self._write_samples("after")
        fixed_after = self._fixed_noise_loss()
        n = len(self.losses)
        tail = self.losses[-max(1, n // 10):] if n else [0.0]
        head = self.losses[:max(1, n // 10)] if n else [0.0]
        return {
            # Der Diffusions-Loss schwankt je nach gezogenem Timestep stark.
            # Das Mittel der letzten 10 % ist aussagekraeftiger als der letzte Wert.
            "final_train_loss": float(sum(tail) / len(tail)),
            "first_train_loss": float(sum(head) / len(head)),
            "total_steps": int(self.global_step),
            "total_epochs": int(self.epochs_done),
            "best_epoch": 0,
            "training_duration_seconds": int(time.time() - self._start_time),
            "architecture": self.pipeline_class or "unbekannt",
            "n_train": len(self.train_dataset),
            "n_val": 0,
            "device": self.device_used,
            "resolution": self.resolution,
            "lora_r": int(self.config.lora_r),
            "num_samples_written": len(self.samples_written),
            # Paar fuer die Analyse-Seite: <key> und <key>_before.
            **({"fixed_noise_loss": fixed_after} if fixed_after is not None else {}),
            **({"fixed_noise_loss_before": self.fixed_loss_before} if self.fixed_loss_before is not None else {}),
        }

    # ── 6. Export ───────────────────────────────────────────────────────────
    def export(self) -> str:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        weights = self._save_lora(out)
        info = {
            "base_model_path": str(Path(self.config.model_path).resolve()),
            "pipeline_class": self.pipeline_class,
            "pipeline_kind": self.kind,
            "resolution": self.resolution,
            "instance_prompt": self.instance_prompt,
            "validation_prompt": self.validation_prompt,
            "lora_r": int(self.config.lora_r),
            "lora_alpha": int(self.config.lora_alpha),
            "target_modules": list(self.config.lora_target_modules or []) or LORA_TARGETS,
            "prediction_type": getattr(self.noise_scheduler.config, "prediction_type", "epsilon"),
            "steps": int(self.global_step),
            "merged": bool(self.merge_lora),
            "samples": [Path(p).name for p in self.samples_written],
        }
        (out / LORA_INFO_FILE).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        MessageProtocol.status("export", f"LoRA-Gewichte gespeichert: {weights.name} "
                                         f"({weights.stat().st_size / 1e6:.1f} MB)")
        if self.merge_lora:
            MessageProtocol.status("export", "Fuehre LoRA ins UNet zusammen und speichere die komplette Pipeline...")
            import torch
            unet = self.pipe.unet
            unet.fuse_lora()
            unet.unload_lora()
            # fp32 speichern, damit Test und Labor unabhaengig vom Trainingsgeraet laden.
            self.pipe.to(dtype=torch.float32)
            self.pipe.save_pretrained(str(out), safe_serialization=True)
            MessageProtocol.status("export", "Vollstaendige Pipeline gespeichert — laedt ohne Basismodell.")
        MessageProtocol.status("export", f"Modell gespeichert: {out}")
        return str(out)
