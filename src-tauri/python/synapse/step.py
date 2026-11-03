"""Ein Lernschritt zum Anschauen: Forward, Loss, Backprop, Update je Node."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .check import guess_input
from .graph import load_graph
from .runtime import SynapseModule

BATCH = 4


def _example(shape: Any, dtype: str) -> Any:
    if isinstance(shape, dict):
        return tuple(_example(s, dtype) for s in (shape["out"], shape["decoder"]))
    full = (BATCH, *tuple(shape)[1:])
    if dtype == "int64":
        return torch.randint(1, 8, full)
    return torch.randn(*full)


def _stats(t: torch.Tensor) -> Dict[str, float]:
    f = t.detach().float()
    return {"mean": round(f.mean().item(), 4), "std": round(f.std().item(), 4) if f.numel() > 1 else 0.0}


def one_step(raw: Dict[str, Any], lr: float = 0.01, seed: int = 0) -> Dict[str, Any]:
    """Zeigt, was ein einziger Trainingsschritt mit Zufallsdaten im Netz bewegt."""
    torch.manual_seed(seed)
    shape, dtype = guess_input(raw)
    graph = load_graph(raw)
    model = SynapseModule(graph, input_shape=shape, input_dtype=getattr(torch, dtype))
    x = _example(shape, dtype)
    values: Dict[Any, torch.Tensor] = {}
    out = model.forward(x, trace=values)
    out = out if isinstance(out, torch.Tensor) else next(iter(out.values()))
    if out.dim() >= 2 and out.size(-1) > 1:
        target = torch.randint(0, out.size(-1), out.shape[:-1])
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), target.reshape(-1))
        loss_name = "cross_entropy (Zufallsziele)"
    else:
        loss = (out.float() ** 2).mean()
        loss_name = "mittleres Quadrat"
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad
    nodes: Dict[str, Any] = {}
    for (nid, port), value in values.items():
        if port == "out" and isinstance(value, torch.Tensor) and nid != "__input__":
            nodes.setdefault(nid, {"shape": list(value.shape), **_stats(value)})
    for name, p in model.named_parameters():
        nid = name.split(".")[1] if name.startswith("layers.") else name
        entry = nodes.setdefault(nid, {})
        grad = 0.0 if p.grad is None else p.grad.norm().item()
        entry["grad"] = round(entry.get("grad", 0.0) + grad, 6)
        entry["update"] = round(entry.get("update", 0.0) + (p.detach() - before[name]).norm().item(), 6)
        entry["params"] = entry.get("params", 0) + p.numel()
    return {"loss": round(loss.item(), 4), "loss_name": loss_name, "lr": lr, "nodes": nodes,
            "order": [n for n in model.order if n in nodes]}
