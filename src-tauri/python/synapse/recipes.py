"""Rezepte: wie aus Modell-Ausgabe und Batch Loss und Metrik werden."""

from __future__ import annotations

from typing import Any, Tuple

import torch

SUPPORTED = ("classification", "regression", "language_model", "seq2seq", "segmentation", "autoencoder", "gan",
             "diffusion", "detection")
ALIASES = {"generation": "language_model"}


def resolve_task(name: str) -> str:
    task = ALIASES.get(name, name)
    if task not in SUPPORTED:
        raise ValueError(f"Task '{name}' wird noch nicht unterstuetzt. Moeglich: {', '.join(SUPPORTED)}")
    return task


def _main(out: Any) -> torch.Tensor:
    return out if isinstance(out, torch.Tensor) else next(iter(out.values()))


def loss_of(task: str, model: Any, out: Any, x: torch.Tensor, y: torch.Tensor, criterion: Any) -> torch.Tensor:
    pred = _main(out)
    if task == "detection":
        from .generative import box_loss
        return box_loss(pred, y)
    if task == "regression":
        loss = criterion(pred.squeeze(-1), y.float())
    elif task in ("language_model", "seq2seq"):
        loss = criterion(pred.reshape(-1, pred.size(-1)), y.reshape(-1))
    elif task == "autoencoder":
        loss = criterion(pred, x)
    elif task == "diffusion":
        loss = criterion(pred, y)
    else:
        loss = criterion(pred, y)
    aux = getattr(model, "aux_loss", None)
    return loss + aux if aux is not None else loss


def prepare(task: str, x: Any, y: Any, plan: Any = None) -> Tuple[Any, Any]:
    """Diffusion trainiert auf verrauschten Bildern, Ziel ist das Rauschen."""
    return plan.add_noise(x) if task == "diffusion" else (x, y)


def score_of(task: str, out: Any, y: torch.Tensor) -> Tuple[int, int]:
    """(richtig, gezaehlt) fuer die Genauigkeit; Padding zaehlt nicht."""
    pred = _main(out)
    if task == "classification":
        return int((pred.argmax(dim=-1) == y).sum()), y.size(0)
    if task == "segmentation":
        return int((pred.argmax(dim=1) == y).sum()), y.numel()
    if task in ("language_model", "seq2seq"):
        keep = y != 0
        return int(((pred.argmax(dim=-1) == y) & keep).sum()), int(keep.sum())
    return 0, 0


def has_accuracy(task: str) -> bool:
    return task in ("classification", "segmentation", "language_model", "seq2seq")
