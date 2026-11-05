"""Laufzeit: fuehrt einen Graphen als nn.Module aus, Werte je (Node, Port)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .graph import ENTRY, Graph

Key = Tuple[str, str]


class SynapseModule(nn.Module):
    """Dasselbe Modul trainiert, prueft und sagt vorher.

    Mit input_shape entstehen Layer in Ausfuehrungsreihenfolge auf Meta-Tensoren,
    dadurch loesen sich 'auto'-Groessen aus der echten Vorstufe auf.
    """

    def __init__(self, graph: Graph, input_shape: Optional[Tuple[int, ...]] = None,
                 input_dtype: torch.dtype = torch.float32, on_node: Any = None):
        super().__init__()
        self.graph = graph
        self.order = graph.order()
        self.node_map = {n.id: n for n in graph.nodes}
        self.incoming: Dict[str, Dict[str, Key]] = {}
        for e in graph.edges:
            self.incoming.setdefault(e.target, {})[e.target_port] = (e.source, e.source_port)

        layers: Dict[str, nn.Module] = {}
        self.input_shapes: Dict[str, Any] = {}
        meta: Dict[Key, torch.Tensor] = {}
        if input_shape is not None:
            if isinstance(input_shape, dict):  # Seq2Seq: Encoder- und Decoder-Eingang
                meta[(ENTRY, "out")] = torch.empty(*input_shape["out"], dtype=input_dtype, device="meta")
                meta[(ENTRY, "decoder")] = torch.empty(*input_shape["decoder"], dtype=input_dtype, device="meta")
            else:
                meta[(ENTRY, "out")] = torch.empty(*input_shape, dtype=input_dtype, device="meta")
        for nid in self.order:
            node = self.node_map[nid]
            spec = node.spec
            if spec.structural:
                continue
            inputs = self._inputs(nid, meta) if meta else {}
            if spec.build is not None:
                shapes = {k: tuple(v.shape) for k, v in inputs.items()} if meta else {}
                self.input_shapes[nid] = shapes.get("in")
                try:
                    layers[nid] = spec.build(node.params, shapes)
                except Exception as e:
                    if on_node:
                        on_node(nid, None, e)
                        break  # Folgefehler verdecken nur die Ursache
                    raise ValueError(f"Node {nid} ({node.type}): {e}") from e
            if meta:
                mod = layers.get(nid)
                try:
                    if mod is not None:
                        mod = _on_meta(mod)
                    out = spec.run(mod, node.params, inputs)
                except Exception as e:
                    if on_node:
                        on_node(nid, None, e)
                        break  # Folgefehler verdecken nur die Ursache
                    raise ValueError(f"Node {nid} ({node.type}): {e}") from e
                for port, value in out.items():
                    meta[(nid, port)] = value
                if on_node:
                    on_node(nid, {k: tuple(v.shape) for k, v in out.items()}, None)
        self.layers = nn.ModuleDict(layers)

    def _inputs(self, nid: str, values: Dict[Key, torch.Tensor]) -> Dict[str, torch.Tensor]:
        node = self.node_map[nid]
        wired = self.incoming.get(nid, {})
        got: Dict[str, torch.Tensor] = {}
        for port in node.spec.inputs:
            if port.id in wired:
                got[port.id] = values[wired[port.id]]
            elif port.fallback and port.fallback in got:
                got[port.id] = got[port.fallback]
            elif port.fallback is None:
                raise ValueError(f"Eingang '{port.id}' von Node {nid} ({node.type}) ist nicht verbunden")
        return got

    def forward(self, x: Any, decoder: Optional[torch.Tensor] = None, trace: Optional[Dict[Key, torch.Tensor]] = None) -> Any:
        if isinstance(x, (tuple, list)):
            x, decoder = x
        values: Dict[Key, torch.Tensor] = {(ENTRY, "out"): x}
        if decoder is not None:
            values[(ENTRY, "decoder")] = decoder
        aux: Optional[torch.Tensor] = None
        for nid in self.order:
            node = self.node_map[nid]
            spec = node.spec
            if spec.structural:
                continue
            out = spec.run(self.layers[nid] if nid in self.layers else None, node.params, self._inputs(nid, values))
            for port, value in out.items():
                values[(nid, port)] = value
            if "kl" in out:
                aux = out["kl"] if aux is None else aux + out["kl"]
        self.aux_loss = aux
        if trace is not None:
            trace.update(values)
        results = {name: values[key] for name, key in self.graph.outputs.items()}
        return next(iter(results.values())) if len(results) == 1 else results


def _on_meta(mod: nn.Module) -> nn.Module:
    from .nodes import copy_to_meta
    return copy_to_meta(mod)
