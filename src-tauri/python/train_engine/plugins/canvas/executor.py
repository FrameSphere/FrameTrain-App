"""DynamicGraphModule — runtime DAG forward from Canvas IR."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ir import CanvasGraphIR, IRNode
from layer_factory import SKIP_TYPES, create_layer, is_layer_node, is_op_node
from ops import run_op


class DynamicGraphModule(nn.Module):
    """
    Runtime nn.Module built from Canvas Graph IR.
    Exposes .layers ModuleDict for compatibility with legacy plugin code.
    """

    def __init__(self, ir: CanvasGraphIR):
        super().__init__()
        self.ir = ir
        self.node_map: Dict[str, IRNode] = ir.node_by_id()
        self.in_edges = ir.in_edges_map()
        self.execution_order = list(ir.execution_order)
        self.output_node_id = self._find_output_node_id()

        layers_dict: Dict[str, nn.Module] = {}
        for node in ir.nodes:
            if is_layer_node(node.type):
                mod = create_layer(node.type, node.params)
                if mod is not None:
                    layers_dict[node.id] = mod
        self.layers = nn.ModuleDict(layers_dict)

    def _find_output_node_id(self) -> Optional[str]:
        arch_ids = [
            nid for nid in self.execution_order
            if self.node_map.get(nid) and self.node_map[nid].type not in SKIP_TYPES
            and self.node_map[nid].category not in ("data", "training")
        ]
        if not arch_ids:
            return None
        out_candidates = [nid for nid in self.execution_order if self.node_map[nid].type == "output_node"]
        if out_candidates:
            return out_candidates[-1]
        sinks = [nid for nid in arch_ids if not any(e.target == nid for e in self.ir.edges if e.source == nid)]
        return sinks[-1] if sinks else arch_ids[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations: Dict[str, torch.Tensor] = {}

        for node_id in self.execution_order:
            node = self.node_map.get(node_id)
            if not node:
                continue

            ntype = node.type
            if ntype in SKIP_TYPES or node.category in ("data", "training"):
                continue

            src_ids = self.in_edges.get(node_id, [])
            if ntype == "input":
                activations[node_id] = x
                continue

            inputs: List[torch.Tensor] = []
            for sid in src_ids:
                if sid in activations:
                    inputs.append(activations[sid])
            if not inputs:
                inputs = [x]

            v0 = inputs[0]
            v1 = inputs[1] if len(inputs) > 1 else v0

            if is_layer_node(ntype):
                out = self._forward_layer(node_id, ntype, v0)
            elif is_op_node(ntype):
                out = run_op(ntype, node.params, inputs)
            else:
                out = v0

            activations[node_id] = out

        if self.output_node_id and self.output_node_id in activations:
            return activations[self.output_node_id]

        for nid in reversed(self.execution_order):
            if nid in activations:
                return activations[nid]
        return x

    def _forward_layer(self, node_id: str, ntype: str, v0: torch.Tensor) -> torch.Tensor:
        layer = self.layers[node_id]

        if ntype == "dense":
            if v0.dim() == 4:
                v0 = v0.mean(dim=(2, 3))
            elif v0.dim() == 3:
                v0 = v0.mean(dim=1)
            return layer(v0)

        if ntype == "embedding":
            return layer(v0.long())

        if ntype == "lstm":
            if v0.dim() == 2:
                v0 = v0.unsqueeze(1)
            out_seq, _ = layer(v0)
            return out_seq[:, -1, :]

        if ntype == "attention":
            if v0.dim() == 2:
                v0 = v0.unsqueeze(1)
            out, _ = layer(v0, v0, v0)
            return out.squeeze(1) if out.dim() == 3 and out.size(1) == 1 else out

        if ntype == "transformer_block":
            if v0.dim() == 2:
                v0 = v0.unsqueeze(1)
            out = layer(v0)
            return out.squeeze(1) if out.dim() == 3 and out.size(1) == 1 else out

        return layer(v0)
