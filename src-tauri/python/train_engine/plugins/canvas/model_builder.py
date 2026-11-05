"""build_model_from_graph — runtime nn.Module from IR."""

from __future__ import annotations

import sys
from pathlib import Path

_CANVAS_DIR = Path(__file__).resolve().parent
for _p in (str(_CANVAS_DIR), str(_CANVAS_DIR.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ir import CanvasGraphIR  # noqa: E402
from synapse.graph import migrate_v1  # noqa: E402
from synapse.runtime import SynapseModule  # noqa: E402


def has_auto_params(ir: CanvasGraphIR) -> bool:
    if ir.training.task_type in ("gan", "diffusion"):
        return True  # Bloecke und Rausch-Plan brauchen die Datenform
    return any(v == "auto" for n in ir.nodes for v in (n.params or {}).values())


def build_model_from_graph(ir: CanvasGraphIR, input_shape=None, input_dtype=None) -> SynapseModule:
    import torch
    if ir.training.task_type == "gan":
        from synapse.generative import GanModel
        latent = next((int(n.params.get("dim", 64)) for n in ir.nodes if n.type == "latent"), 64)
        return GanModel(ir.nodes, latent, input_shape)
    graph = migrate_v1(ir.nodes, ir.edges, ir.execution_order)
    return SynapseModule(graph, input_shape=input_shape, input_dtype=input_dtype or torch.float32)
