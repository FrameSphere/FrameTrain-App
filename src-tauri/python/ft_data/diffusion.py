"""Diffusion-Pipelines laden und Bilder erzeugen — fuer Training, Test und Labor.

Ein trainiertes text_to_image_lora-Modell liegt in einer von zwei Formen vor:
  * nur LoRA:  pytorch_lora_weights.safetensors + text_to_image_lora.json
               (die JSON nennt den Pfad des Basismodells; ein paar MB statt GB)
  * gemergt:   vollstaendige diffusers-Pipeline (model_index.json + Unterordner),
               zusaetzlich die LoRA-Datei daneben. Laedt ohne Basismodell.
Beide laden hier ueber dieselbe Funktion, damit Test-Plugin und Modell-Server
nicht auseinanderlaufen.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

LORA_INFO_FILE = "text_to_image_lora.json"
LORA_WEIGHTS_FILE = "pytorch_lora_weights.safetensors"

# Pipeline-Klasse aus model_index.json -> Variante. SD 1.x und 2.x teilen die
# Klasse; 2.x unterscheidet sich nur in prediction_type (v_prediction), das
# der Scheduler mitbringt.
SUPPORTED_PIPELINES = {
    "StableDiffusionPipeline": "sd",
    "StableDiffusionXLPipeline": "sdxl",
}


def read_model_index(path: Path) -> Dict[str, Any]:
    f = Path(path) / "model_index.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def read_lora_info(path: Path) -> Dict[str, Any]:
    f = Path(path) / LORA_INFO_FILE
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def is_diffusion_dir(path: Path) -> bool:
    p = Path(path)
    return (p / "model_index.json").exists() or (p / LORA_INFO_FILE).exists()


def pipeline_kind(class_name: str) -> Optional[str]:
    return SUPPORTED_PIPELINES.get(str(class_name or ""))


def unsupported_message(class_name: str) -> str:
    return (
        f"Die Pipeline '{class_name or 'unbekannt'}' wird fuer LoRA-Training nicht unterstuetzt. "
        "Unterstuetzt sind Stable Diffusion 1.x/2.x (StableDiffusionPipeline) und "
        "SDXL (StableDiffusionXLPipeline). SD3, FLUX, PixArt, Kandinsky und Bild-zu-Bild-"
        "Pipelines haben einen anderen Aufbau (Transformer statt UNet)."
    )


def torch_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pipeline_class(kind: str):
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    return StableDiffusionXLPipeline if kind == "sdxl" else StableDiffusionPipeline


def weight_variant(model_dir: Path) -> Optional[str]:
    """"fp16", wenn die Komponenten NUR als fp16-Variante vorliegen.

    Viele Repos (und der Import der App, wenn es keine vollen Gewichte gibt)
    liefern diffusion_pytorch_model.fp16.safetensors / model.fp16.safetensors.
    diffusers laedt die ohne variant="fp16" gar nicht ("no file named ...").
    Gerechnet wird trotzdem in dem dtype, der beim Laden verlangt wird.
    """
    model_dir = Path(model_dir)
    has_full, has_fp16 = False, False
    for comp in ("unet", "transformer", "vae", "text_encoder"):
        d = model_dir / comp
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix not in (".safetensors", ".bin"):
                continue
            if ".fp16." in f.name:
                has_fp16 = True
            elif not any(t in f.name for t in (".bf16.", ".ema.", ".non_ema.")):
                has_full = True
    return "fp16" if has_fp16 and not has_full else None


def load_base_pipeline(model_dir: Path, dtype=None):
    """Laedt eine diffusers-Pipeline aus einem Ordner mit model_index.json.

    Der Safety-Checker wird nicht geladen: er kostet ~1 GB Speicher und
    schwaerzt bei feinjustierten Stilen haeufig harmlose Bilder. Die Pipeline
    laeuft lokal fuer den Nutzer, der das Modell selbst gewaehlt hat.
    """
    import torch

    index = read_model_index(model_dir)
    cls_name = index.get("_class_name", "")
    kind = pipeline_kind(cls_name)
    if kind is None:
        raise ValueError(unsupported_message(cls_name))
    cls = _pipeline_class(kind)
    kwargs: Dict[str, Any] = {"torch_dtype": dtype or torch.float32}
    if "safety_checker" in index:
        kwargs.update(safety_checker=None, requires_safety_checker=False)
    variant = weight_variant(Path(model_dir))
    if variant:
        kwargs["variant"] = variant
    pipe = cls.from_pretrained(str(model_dir), **kwargs)
    pipe.set_progress_bar_config(disable=True)
    return pipe, kind


def load_pipeline(model_path: Path, device=None, status: Optional[Callable[[str], None]] = None):
    """Laedt ein trainiertes (oder unveraendertes) Modell fuer die Bilderzeugung.

    Rueckgabe: (pipeline, info). info enthaelt die Trainings-Metadaten
    (Aufloesung, Prompts), soweit vorhanden.
    """
    import torch

    model_path = Path(model_path)
    device = device or torch_device()
    info = read_lora_info(model_path)
    # fp16 nur auf CUDA. Auf MPS erzeugt der VAE in fp16 NaNs (schwarze Bilder),
    # auf der CPU ist fp16 langsamer als fp32.
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if (model_path / "model_index.json").exists():
        if status:
            status("Lade Diffusion-Pipeline...")
        pipe, kind = load_base_pipeline(model_path, dtype)
    elif info:
        base = Path(str(info.get("base_model_path", "")))
        if not (base / "model_index.json").exists():
            raise FileNotFoundError(
                f"Basismodell fuer die LoRA-Gewichte nicht gefunden: {base}\n"
                "Die LoRA-Datei braucht das Modell, auf dem sie trainiert wurde. "
                "Liegt es nicht mehr dort, das Training mit plugin_config merge_lora=true "
                "wiederholen — dann entsteht eine eigenstaendige Pipeline."
            )
        if status:
            status(f"Lade Basismodell {base.name} und LoRA-Gewichte...")
        pipe, kind = load_base_pipeline(base, dtype)
        if not (model_path / LORA_WEIGHTS_FILE).exists():
            raise FileNotFoundError(f"{LORA_WEIGHTS_FILE} fehlt in {model_path}")
        pipe.load_lora_weights(str(model_path), weight_name=LORA_WEIGHTS_FILE)
    else:
        raise FileNotFoundError(
            f"Kein Diffusion-Modell in {model_path}: erwartet model_index.json "
            f"(Pipeline) oder {LORA_INFO_FILE} + {LORA_WEIGHTS_FILE} (LoRA)."
        )
    pipe.to(device)
    info = dict(info)
    info.setdefault("pipeline_kind", kind)
    return pipe, info


def default_size(pipe, info: Dict[str, Any]) -> int:
    """Bildgroesse: die Trainingsaufloesung, sonst die native des UNet."""
    res = int(info.get("resolution") or 0)
    if res > 0:
        return res
    try:
        return int(pipe.unet.config.sample_size) * int(getattr(pipe, "vae_scale_factor", 8))
    except Exception:
        return 512


def generate_image(pipe, prompt: str, *, steps: int = 25, guidance: float = 7.5,
                   seed: Optional[int] = None, negative_prompt: str = "",
                   size: Optional[int] = None):
    """Ein Bild zu einem Prompt. Der Generator liegt auf der CPU — MPS-Generatoren
    liefern je nach torch-Version andere Zahlen und sind nicht reproduzierbar."""
    import torch

    kwargs: Dict[str, Any] = dict(
        prompt=prompt, num_inference_steps=max(1, int(steps)), guidance_scale=float(guidance),
    )
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if size:
        kwargs["height"] = kwargs["width"] = int(size)
    if seed is not None:
        kwargs["generator"] = torch.Generator(device="cpu").manual_seed(int(seed))
    return pipe(**kwargs).images[0]


def safe_filename(text: str, limit: int = 40) -> str:
    import re
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:limit]
    return cleaned or "bild"
