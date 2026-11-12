"""
CLIP-Score: wie gut passt ein erzeugtes Bild zu seinem Prompt?

CLIP-Score = max(100 * cos(Bild-Embedding, Text-Embedding), 0) — dieselbe
Skala wie torchmetrics.multimodal.CLIPScore (typisch 20..35 bei SD 1.5 mit
clip-vit-base-patch32; hoeher = naeher am Prompt).

Das CLIP-Modell wird nur lokal gesucht (Hugging-Face-Cache oder ein Pfad aus
plugin_config "clip_model"). Ein Test laedt nie ungefragt Hunderte MB nach;
fehlt das Modell, laeuft der Test ohne Kennzahl weiter und nennt den Grund.
"""
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def clip_score_from_embeddings(image_emb: Sequence[float], text_emb: Sequence[float]) -> float:
    """100 * Kosinus, nach unten bei 0 begrenzt."""
    dot = sum(a * b for a, b in zip(image_emb, text_emb))
    ni = sum(a * a for a in image_emb) ** 0.5
    nt = sum(b * b for b in text_emb) ** 0.5
    if not ni or not nt:
        return 0.0
    return max(100.0 * dot / (ni * nt), 0.0)


def mean(values: List[Optional[float]]) -> Optional[float]:
    vs = [v for v in values if v is not None]
    return sum(vs) / len(vs) if vs else None


class ClipScorer:
    def __init__(self, model, processor, device, name: str):
        self.model, self.processor, self.device, self.name = model, processor, device, name

    @classmethod
    def load(cls, name: str = DEFAULT_CLIP_MODEL, device=None) -> Tuple[Optional["ClipScorer"], Optional[str]]:
        """(Scorer, None) oder (None, Grund). Laedt nur lokal vorhandene Modelle."""
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:
            return None, f"CLIP-Score nicht berechnet: Paket fehlt ({e})."
        name = str(name or DEFAULT_CLIP_MODEL)
        local = Path(name).expanduser()
        source = str(local) if local.exists() else name
        try:
            model = CLIPModel.from_pretrained(source, local_files_only=True)
            processor = CLIPProcessor.from_pretrained(source, local_files_only=True)
        except Exception as e:  # nicht im Cache, kaputt, kein CLIP
            return None, (
                f"CLIP-Score nicht berechnet: CLIP-Modell '{name}' ist lokal nicht vorhanden "
                f"({type(e).__name__}). In FrameTrain unter Modelle herunterladen oder "
                "plugin_config clip_model auf einen lokalen Ordner setzen."
            )
        if device is None:
            device = torch.device("cpu")
        model.eval().to(device)
        return cls(model, processor, device, name), None

    def score(self, image, prompt: str) -> float:
        import torch
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt",
                                padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            img = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            txt = self.model.get_text_features(input_ids=inputs["input_ids"],
                                               attention_mask=inputs.get("attention_mask"))
        # transformers 5 kann ein Ausgabeobjekt statt Tensor liefern.
        img = getattr(img, "pooler_output", img)
        txt = getattr(txt, "pooler_output", txt)
        return clip_score_from_embeddings(img[0].float().cpu().tolist(), txt[0].float().cpu().tolist())


def round_or_none(v: Optional[Any], digits: int = 3) -> Optional[float]:
    return None if v is None else round(float(v), digits)
