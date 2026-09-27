"""Vision-Language-Modelle: Prompt bauen, Labels maskieren, Antworten erzeugen.

Training, Test-Plugin und Modell-Server nutzen dieselben Funktionen — sonst
sieht das Modell im Test einen anderen Prompt als im Training und antwortet
schlechter, ohne dass jemand den Grund findet.

Drei Familien:
  * Chat-Template (SmolVLM/Idefics3, Qwen2-VL, LLaVA ...): Nutzer-Zug mit Bild
    und Frage, Assistenten-Zug mit der Antwort.
  * PaliGemma: Frage als Praefix, Antwort als `suffix` — der Processor baut
    die Labels selbst.
  * ohne Template (BLIP, Florence-2): Frage als Text-Praefix, Antwort dahinter;
    Encoder-Decoder-Modelle bekommen die Antwort als eigene Labels.
Geloss wird nur auf die Antwort: alles, was Prompt und Antwort gemeinsam am
Anfang haben (Bild-Tokens, Systemtext, Frage), steht in den Labels auf -100.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

VLM_INFO_FILE = "vlm_info.json"
DEFAULT_PROMPT = "Beschreibe das Bild."

# model_type-Werte, die AutoModelForImageTextToText laedt und die hier gemeint sind.
VLM_MODEL_TYPES = {
    "idefics3", "smolvlm", "idefics2", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "paligemma",
    "blip", "blip-2", "blip_2", "llava", "llava_next", "llava_onevision", "florence2",
    "gemma3", "mllama", "pixtral", "aya_vision", "internvl", "kosmos-2", "git",
}

# Teile des Modells, die zum Bild-Encoder oder zur Bruecke gehoeren.
VISION_MARKERS = (
    "vision", "visual", "image_tower", "vision_tower", "vision_model", "patch_embed",
    "connector", "multi_modal_projector", "mm_projector", "image_encoder", "img_",
)

# Kandidaten fuer LoRA, in Gruppen nach Vorrang.
LORA_GROUPS = (
    ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj"),
    ("query", "key", "value"),
)


def is_vlm_config(cfg: Dict[str, Any]) -> bool:
    mt = str(cfg.get("model_type", "")).lower()
    archs = [a for a in (cfg.get("architectures") or []) if isinstance(a, str)]
    if mt in VLM_MODEL_TYPES:
        return True
    return any(a.endswith(("ForVision2Seq", "ForImageTextToText")) for a in archs) or (
        "vision_config" in cfg and any(a.endswith("ForConditionalGeneration") for a in archs))


def read_info(model_path: Path) -> Dict[str, Any]:
    f = Path(model_path) / VLM_INFO_FILE
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def find_lora_targets(model, train_vision: bool = False) -> List[str]:
    """Vollstaendige Namen der Linear-Schichten fuer LoRA.

    Volle Namen statt Kurzformen: "q_proj" allein traefe bei SmolVLM auch den
    Bild-Encoder (dessen Attention heisst genauso) — der soll eingefroren bleiben.
    """
    import torch.nn as nn

    linear = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for group in LORA_GROUPS:
        names = [n for n, _ in linear
                 if n.split(".")[-1] in group
                 and (train_vision or not any(v in n.lower() for v in VISION_MARKERS))]
        if names:
            return names
    raise ValueError(
        "Keine passenden Attention-Projektionen fuer LoRA gefunden (q_proj/k_proj/v_proj/o_proj "
        "bzw. query/key/value). Setze lora_target_modules im Training manuell.")


def configure_processor(processor, image_splitting: Optional[bool] = None, max_pixels: int = 0) -> None:
    """Bildaufloesung begrenzen — sonst wird ein Bild bei SmolVLM zu 17 Kacheln
    (>1000 Tokens) und das Training auf dem Mac zaeh."""
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return
    if image_splitting is not None and hasattr(ip, "do_image_splitting"):
        ip.do_image_splitting = bool(image_splitting)
    if max_pixels and max_pixels > 0 and hasattr(ip, "max_pixels"):
        ip.max_pixels = int(max_pixels)
        if hasattr(ip, "size") and isinstance(ip.size, dict) and "longest_edge" in ip.size:
            ip.size["longest_edge"] = int(max_pixels)


def uses_chat_template(processor) -> bool:
    return bool(getattr(processor, "chat_template", None)) or bool(
        getattr(getattr(processor, "tokenizer", None), "chat_template", None) and
        getattr(processor, "image_token", None))


def family(model_type: str, processor, is_encoder_decoder: bool = False) -> str:
    mt = (model_type or "").lower()
    if mt == "paligemma":
        return "paligemma"
    if is_encoder_decoder or mt == "florence2":
        return "encdec"
    if uses_chat_template(processor):
        return "chat"
    return "plain"


def _user_turn(question: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "image"}]
    if question:
        content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def prompt_text(processor, fam: str, question: str) -> str:
    if fam == "chat":
        return processor.apply_chat_template(_user_turn(question), add_generation_prompt=True)
    return question


def full_text(processor, fam: str, question: str, answer: str) -> str:
    if fam == "chat":
        msgs = _user_turn(question) + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        return processor.apply_chat_template(msgs)
    if fam == "plain":
        return f"{question} {answer}".strip() if question else answer
    return question


def _images_arg(fam: str, images: Sequence[Any]):
    # Chat-Processors (Idefics3 & Co.) erwarten je Beispiel eine Bildliste.
    return [[im] for im in images] if fam == "chat" else list(images)


def _common_prefix(a, b) -> int:
    n = min(len(a), len(b))
    eq = (a[:n] == b[:n]).tolist()
    for i, same in enumerate(eq):
        if not same:
            return i
    return n


def collate(processor, fam: str, images: Sequence[Any], questions: Sequence[str],
            answers: Sequence[str], max_length: int = 0) -> Dict[str, Any]:
    """Batch mit Labels nur auf den Antwort-Tokens."""
    tok = getattr(processor, "tokenizer", processor)
    tok.padding_side = "right"
    imgs = _images_arg(fam, images)

    if fam == "paligemma":
        return processor(text=list(questions), images=imgs, suffix=list(answers),
                         return_tensors="pt", padding="longest")
    if fam == "encdec":
        enc = processor(text=list(questions), images=imgs, return_tensors="pt", padding=True)
        lab = tok(list(answers), return_tensors="pt", padding=True).input_ids
        lab[lab == tok.pad_token_id] = -100
        enc["labels"] = lab
        return enc

    fulls = [full_text(processor, fam, q, a) for q, a in zip(questions, answers)]
    if fam == "plain" and getattr(tok, "eos_token", None) and not getattr(tok, "sep_token", None):
        fulls = [f + tok.eos_token for f in fulls]
    enc = processor(text=fulls, images=imgs, return_tensors="pt", padding=True)
    prompts = [prompt_text(processor, fam, q) for q in questions]
    penc = processor(text=prompts, images=imgs, return_tensors="pt", padding=True)
    labels = enc["input_ids"].clone()
    for i in range(labels.shape[0]):
        plen = int(penc["attention_mask"][i].sum())
        cut = _common_prefix(penc["input_ids"][i, :plen], enc["input_ids"][i])
        # Mindestens ein Antwort-Token muss uebrig bleiben, sonst ist der Loss NaN.
        n_real = int(enc["attention_mask"][i].sum())
        labels[i, :min(cut, max(n_real - 1, 0))] = -100
    labels[enc["attention_mask"] == 0] = -100
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is not None:
        labels[labels == image_token_id] = -100
    enc["labels"] = labels
    if max_length and enc["input_ids"].shape[1] > max_length:
        for k in ("input_ids", "attention_mask", "labels"):
            enc[k] = enc[k][:, :max_length]
    return enc


def generate(model, processor, fam: str, images: Sequence[Any], questions: Sequence[str],
             max_new_tokens: int = 64) -> List[str]:
    """Antworten fuer einen Batch (gierig, ohne Sampling — reproduzierbar)."""
    import torch

    tok = getattr(processor, "tokenizer", processor)
    tok.padding_side = "left"
    imgs = _images_arg(fam, images)
    prompts = [prompt_text(processor, fam, q) for q in questions]
    kwargs: Dict[str, Any] = dict(text=prompts, images=imgs, return_tensors="pt", padding=True)
    if fam == "plain" and not any(prompts):
        kwargs.pop("text")
    enc = processor(**kwargs)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    enc = {k: (v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device))
           if hasattr(v, "to") else v for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=int(max_new_tokens), do_sample=False)
    tok.padding_side = "right"
    ids = enc.get("input_ids")
    results = []
    for i in range(out.shape[0]):
        seq = out[i]
        if ids is not None and fam in ("chat", "paligemma") and seq.shape[0] >= ids.shape[1] \
                and bool((seq[:ids.shape[1]] == ids[i]).all()):
            seq = seq[ids.shape[1]:]
        text = tok.decode(seq, skip_special_tokens=True).strip()
        q = questions[i].strip()
        if fam == "plain" and q and text.lower().startswith(q.lower()):
            text = text[len(q):].strip()
        results.append(text)
    return results


def load_model(model_path: Path, device, dtype=None):
    """Modell + Processor fuer Inferenz (Test-Plugin, Modell-Server)."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    model_path = Path(model_path)
    processor = AutoProcessor.from_pretrained(str(model_path))
    cfg = AutoConfig.from_pretrained(str(model_path))
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), dtype=dtype or torch.float32)
    model.to(device).eval()
    fam = family(getattr(cfg, "model_type", ""), processor,
                 bool(getattr(cfg, "is_encoder_decoder", False)))
    return model, processor, fam, read_info(model_path)
