"""GAN, Diffusion und Detektion: Rezepte mit eigener Rechnung."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import split_indices, split_settings


# ── GAN ──────────────────────────────────────────────────────────────────────
class GanModel(nn.Module):
    """Zwei Bloecke im Graph: Rolle generator und discriminator."""

    def __init__(self, nodes: Sequence[Any], latent: int, shape: Tuple[int, ...]):
        super().__init__()
        from .graph import parse_block
        from .runtime import SynapseModule
        roles = {str((n.params or {}).get("role", "")): n for n in nodes if n.type == "block"}
        missing = [r for r in ("generator", "discriminator") if r not in roles]
        if missing:
            raise ValueError(f"GAN braucht je einen Block mit Rolle generator und discriminator, es fehlt: {missing}")
        self.latent = latent
        self.generator = SynapseModule(parse_block(roles["generator"].params["graph"]), input_shape=(1, latent))
        self.discriminator = SynapseModule(parse_block(roles["discriminator"].params["graph"]), input_shape=shape)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)


def gan_epoch(model: GanModel, loader: DataLoader, opt_g: Any, opt_d: Any, device: str, stopped: Any) -> Tuple[float, float]:
    """Abwechselnd: Diskriminator lernt echt von falsch, Generator lernt zu taeuschen."""
    model.train()
    bce = F.binary_cross_entropy_with_logits
    g_sum = d_sum = 0.0
    steps = 0
    for real, _ in loader:
        if stopped():
            break
        real = real.to(device)
        z = torch.randn(real.size(0), model.latent, device=device)
        fake = model.generator(z)
        real_score = model.discriminator(real)
        fake_score = model.discriminator(fake.detach())
        d_loss = bce(real_score, torch.ones_like(real_score)) + bce(fake_score, torch.zeros_like(fake_score))
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        fooled = model.discriminator(model.generator(z))
        g_loss = bce(fooled, torch.ones_like(fooled))
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        g_sum += g_loss.item()
        d_sum += d_loss.item()
        steps += 1
    return g_sum / max(steps, 1), d_sum / max(steps, 1)


# ── Diffusion ────────────────────────────────────────────────────────────────
class NoisePlan:
    """Rausch-Plan: wie viel vom Bild in Schritt t noch uebrig ist."""

    def __init__(self, steps: int = 100):
        self.steps = steps
        self.beta = torch.linspace(1e-4, 0.1, steps)
        self.alpha = 1 - self.beta
        self.bar = torch.cumprod(self.alpha, 0)

    def add_noise(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        t = torch.randint(0, self.steps, (x.size(0),), device=x.device)
        noise = torch.randn_like(x)
        keep = self.bar.to(x.device)[t].view(-1, *[1] * (x.dim() - 1))
        return (keep.sqrt() * x + (1 - keep).sqrt() * noise, (t.float() / self.steps).unsqueeze(1)), noise

    @torch.no_grad()
    def sample(self, model: Any, shape: Sequence[int], device: str) -> torch.Tensor:
        x = torch.randn(*shape, device=device)
        for i in reversed(range(self.steps)):
            t = torch.full((shape[0], 1), i / self.steps, device=device)
            eps = model(x, t)
            a, b, bar = self.alpha[i].item(), self.beta[i].item(), self.bar[i].item()
            prev = self.bar[i - 1].item() if i > 0 else 1.0
            # Das Bild wird je Schritt begrenzt, sonst schaukeln sich Fehler in den
            # verrauschten Schritten auf.
            start = ((x - math.sqrt(1 - bar) * eps) / math.sqrt(bar)).clamp(-3, 3)
            x = (math.sqrt(prev) * b * start + math.sqrt(a) * (1 - prev) * x) / (1 - bar)
            if i > 0:
                x = x + math.sqrt(b * (1 - prev) / (1 - bar)) * torch.randn_like(x)
        return x


# ── Detektion ────────────────────────────────────────────────────────────────
class _Boxes(Dataset):
    """Bilder mit YOLO-Labels als Gitter [5 + Klassen, S, S]: Objekt, x, y, w, h, Klasse."""

    def __init__(self, items: List[Tuple[Path, Path]], size: int, grid: int, classes: int):
        self.items, self.size, self.grid, self.classes = items, size, grid, classes

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image
        image, label = self.items[i]
        with Image.open(image) as im:
            arr = np.asarray(im.convert("RGB").resize((self.size, self.size)), dtype="float32") / 255.0
        x = (torch.from_numpy(arr).permute(2, 0, 1) - 0.5) / 0.5
        y = torch.zeros(5 + self.classes, self.grid, self.grid)
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c, cx, cy, w, h = int(float(parts[0])), *map(float, parts[1:5])
                gx, gy = min(int(cx * self.grid), self.grid - 1), min(int(cy * self.grid), self.grid - 1)
                y[:5, gy, gx] = torch.tensor([1.0, cx * self.grid - gx, cy * self.grid - gy, w, h])
                y[5 + c, gy, gx] = 1.0
        return x, y


def box_loaders(root: str, params: Dict[str, Any], pipeline: Sequence[Dict[str, Any]], batch_size: int,
                num_classes: int, preprocess: Dict[str, Any], status) -> Tuple[DataLoader, DataLoader]:
    from ft_data.media import IMAGE_EXTS
    base = Path(root)
    img_dir = base / "images" / "train" if (base / "images" / "train").is_dir() else base / "images"
    lab_dir = base / "labels" / "train" if (base / "labels" / "train").is_dir() else base / "labels"
    items = [(f, lab_dir / f"{f.stem}.txt") for f in sorted(img_dir.rglob("*")) if f.suffix.lower() in IMAGE_EXTS]
    if not items:
        raise ValueError(f"detection_loader: Keine Bilder in {img_dir}. Erwartet: images/ und labels/ im YOLO-Format.")
    size, grid = int(params.get("imageSize", 128)), int(params.get("gridSize", 8))
    classes = int(num_classes)
    tr, va = split_indices(len(items), split_settings(pipeline))
    preprocess.update(pipeline={"kind": "boxes", "size": size, "grid": grid, "classes": classes}, class_names=[])
    status(f"detection_loader: {len(items)} Bilder | Gitter {grid} | Train: {len(tr)} | Val: {len(va)}")
    gen = torch.Generator().manual_seed(42)
    return (DataLoader(_Boxes([items[i] for i in tr], size, grid, classes), batch_size=batch_size, shuffle=True, generator=gen),
            DataLoader(_Boxes([items[i] for i in va], size, grid, classes), batch_size=batch_size))


def box_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    obj = y[:, 0] > 0.5
    loss = F.binary_cross_entropy_with_logits(pred[:, 0], y[:, 0])
    if obj.any():
        boxes = F.mse_loss(torch.sigmoid(pred[:, 1:5]).permute(0, 2, 3, 1)[obj], y[:, 1:5].permute(0, 2, 3, 1)[obj])
        loss = loss + 5 * boxes
        if y.size(1) > 5:
            loss = loss + F.cross_entropy(pred[:, 5:].permute(0, 2, 3, 1)[obj], y[:, 5:].argmax(1)[obj])
    return loss


def decode_boxes(pred: torch.Tensor, threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Gitter zurueck in Boxen (normiert 0 bis 1, Mitte und Groesse) fuer ein Bild."""
    p = pred[0]
    grid = p.size(-1)
    out = []
    score = torch.sigmoid(p[0])
    for gy, gx in (score > threshold).nonzero().tolist():
        cx, cy, w, h = torch.sigmoid(p[1:5, gy, gx]).tolist()
        cls = int(p[5:, gy, gx].argmax()) if p.size(0) > 5 else 0
        out.append({"class": cls, "score": float(score[gy, gx]), "box": [(gx + cx) / grid, (gy + cy) / grid, w, h]})
    return sorted(out, key=lambda b: -b["score"])


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)
