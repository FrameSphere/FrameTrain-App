"""Losses und Optimizer aus der Trainings-Spezifikation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma, self.reduction = gamma, reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        loss = (1 - torch.exp(-ce)) ** self.gamma * ce
        return loss.sum() if self.reduction == "sum" else loss.mean()


class KLDivLoss(nn.Module):
    """KL-Divergenz zwischen Vorhersage und Ziel-Klasse."""

    def __init__(self, smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.smoothing, self.reduction = smoothing, reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        k = logits.size(-1)
        soft = F.one_hot(target, k).float() * (1 - self.smoothing) + self.smoothing / k
        loss = F.kl_div(F.log_softmax(logits, dim=-1), soft, reduction="sum")
        return loss if self.reduction == "sum" else loss / logits.size(0)


class OneHotLoss(nn.Module):
    """Regressions-Loss fuer Klassen-Ziele: vergleicht mit dem One-Hot-Vektor."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.size(-1) == 1:
            return self.inner(logits.squeeze(-1), target.float())
        return self.inner(logits, F.one_hot(target, logits.size(-1)).float())


def make_loss(name: str, regression: bool, reduction: str, smoothing: float, task: str = "classification") -> nn.Module:
    plain = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": nn.HuberLoss, "bce": nn.BCEWithLogitsLoss}
    if task == "detection":
        return nn.Identity()
    if task in ("autoencoder", "diffusion"):
        if name not in plain:
            raise ValueError(f"Loss '{name}' passt nicht zum Autoencoder. Nutze mse, mae, huber oder bce.")
        return plain[name](reduction=reduction)
    if task in ("language_model", "seq2seq", "segmentation") and name in plain:
        raise ValueError(f"Loss '{name}' passt nicht zu {task}. Nutze cross_entropy oder focal.")
    if task in ("language_model", "seq2seq") and name == "cross_entropy":
        return nn.CrossEntropyLoss(reduction=reduction, label_smoothing=smoothing, ignore_index=0)
    if name in plain:
        inner = plain[name](reduction=reduction)
        return inner if regression else OneHotLoss(inner)
    if regression:
        raise ValueError(f"Loss '{name}' passt nicht zu Regression. Nutze mse, mae oder huber.")
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(reduction=reduction, label_smoothing=smoothing)
    if name == "nll":
        return nn.NLLLoss(reduction=reduction)
    if name == "focal":
        return FocalLoss(reduction=reduction)
    if name == "kl_div":
        return KLDivLoss(smoothing, reduction)
    raise ValueError(f"Unbekannter Loss: {name}")


def make_optimizer(name: str, params, lr: float, wd: float) -> optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=wd)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=wd)
    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor
        except ImportError as e:
            from ft_data.deps import missing
            raise missing("transformers", what="Adafactor") from e
        return Adafactor(params, lr=lr, weight_decay=wd, scale_parameter=False, relative_step=False, warmup_init=False)
    raise ValueError(f"Unbekannter Optimizer: {name}")
