"""Graph-Format v2 und die Hebung gespeicherter v1-Graphen."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import nodes  # noqa: F401  (fuellt REGISTRY)
from .spec import REGISTRY, NodeSpec

ENTRY = "__input__"  # Quelle aller Kanten, die aus den Daten kommen


@dataclass
class GNode:
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> NodeSpec:
        return REGISTRY[self.type]


@dataclass(frozen=True)
class GEdge:
    source: str
    source_port: str
    target: str
    target_port: str


@dataclass
class Graph:
    nodes: List[GNode]
    edges: List[GEdge]
    outputs: Dict[str, Tuple[str, str]]  # Name -> (Node, Port)

    def order(self) -> List[str]:
        """Topologische Reihenfolge; Kreise sind ein Fehler."""
        ids = [n.id for n in self.nodes]
        degree = {i: 0 for i in ids}
        after: Dict[str, List[str]] = {i: [] for i in ids}
        for e in self.edges:
            if e.source != ENTRY:
                degree[e.target] += 1
                after[e.source].append(e.target)
        ready = [i for i in ids if degree[i] == 0]
        out: List[str] = []
        while ready:
            cur = ready.pop(0)
            out.append(cur)
            for nb in after[cur]:
                degree[nb] -= 1
                if degree[nb] == 0:
                    ready.append(nb)
        if len(out) != len(ids):
            stuck = [i for i in ids if i not in out]
            raise ValueError(f"Der Graph enthaelt einen Kreis: {', '.join(stuck)}")
        return out


def _value(raw: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    return next((raw[k] for k in keys if raw.get(k) is not None), default)


def _attr(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def migrate_v1(nodes_raw: List[Any], edges_raw: List[Any], order: Optional[List[str]] = None) -> Graph:
    """v1 kannte Kanten ohne sichere Ports und Eingaben ohne Kante.

    Bekannte Fehler des alten Interpreters (Skip vom Input, Split, Output-Node)
    werden dabei behoben. Gewichts-Namen und das Mitteln vor Dense bleiben, damit
    gespeicherte Modelle dieselben Werte liefern.
    """
    known = {_attr(n, "id"): n for n in nodes_raw}
    missing = sorted({_attr(n, "type") for n in nodes_raw if _attr(n, "type") not in REGISTRY})
    if missing:
        raise ValueError(f"Unbekannte Node-Typen: {', '.join(missing)}")

    ids = [i for i in (order or []) if i in known] + [i for i in known if i not in (order or [])]
    gnodes = [GNode(str(i), _attr(known[i], "type"), dict(_attr(known[i], "params") or {})) for i in ids]
    by_id = {n.id: n for n in gnodes}
    for g in gnodes:
        if g.type == "dense":
            g.params.setdefault("implicitPool", True)

    def is_source_of_data(nid: str) -> bool:
        return by_id[nid].spec.family == "data"

    edges: List[GEdge] = []
    taken: Dict[str, set] = {}
    outputs: Dict[str, Tuple[str, str]] = {}
    for raw in edges_raw:
        src, tgt = str(_attr(raw, "source")), str(_attr(raw, "target"))
        if src not in by_id or tgt not in by_id:
            continue
        s, t = by_id[src], by_id[tgt]
        if s.spec.structural and s.spec.family == "training":
            continue
        if t.spec.structural and t.type != "output_node":
            continue  # Daten- und Lern-Nodes haben im forward keinen Eingang
        sh = _attr(raw, "source_handle") if not isinstance(raw, dict) else raw.get("sourceHandle")
        th = _attr(raw, "target_handle") if not isinstance(raw, dict) else raw.get("targetHandle")
        out_port = sh if sh in [p.id for p in s.spec.outputs] else (s.spec.outputs[0].id if s.spec.outputs else "out")
        if t.type == "output_node":
            outputs.setdefault("out", (ENTRY, "out") if is_source_of_data(src) else (src, out_port))
            continue
        names = [p.id for p in t.spec.inputs]
        used = taken.setdefault(tgt, set())
        port = th if th in names and th not in used else next((n for n in names if n not in used), None)
        if port is None:
            continue  # v1 nahm von mehreren Kanten auf einen Port nur die ersten
        used.add(port)
        entry = "decoder" if s.type in ("decoder_input", "timestep") else "out"
        edges.append(GEdge(ENTRY, entry, tgt, port) if is_source_of_data(src) else GEdge(src, out_port, tgt, port))

    # Layer ohne Eingang bekamen in v1 die Modell-Eingabe.
    for g in gnodes:
        if g.spec.structural:
            continue
        if not taken.get(g.id) and g.spec.inputs:
            edges.append(GEdge(ENTRY, "out", g.id, g.spec.inputs[0].id))

    if "out" not in outputs:
        last = next((g for g in reversed(gnodes) if not g.spec.structural), None)
        if last is not None:
            outputs["out"] = (last.id, last.spec.outputs[0].id)
    return Graph(gnodes, edges, outputs)


def parse_v2(raw: Dict[str, Any]) -> Graph:
    model = raw.get("model") or {}
    gnodes = [GNode(str(n["id"]), n["type"], dict(n.get("params") or {})) for n in model.get("nodes", [])]
    edges = [GEdge(str(e["source"]), e.get("sourcePort", "out"), str(e["target"]), e.get("targetPort", "in"))
             for e in model.get("edges", [])]
    outputs = {k: (v["node"], v.get("port", "out")) for k, v in (model.get("outputs") or {}).items()}
    return Graph(gnodes, edges, outputs)


def parse_block(raw: Dict[str, Any]) -> Graph:
    """Block-Inhalt: Layer und Kanten, Eingang ist der Block-Eingang."""
    return migrate_v1(raw.get("nodes", []), raw.get("edges", []), raw.get("execution_order"))


def load_graph(raw: Dict[str, Any]) -> Graph:
    version = raw.get("version")
    if version == 2:
        return parse_v2(raw)
    if version == 1:
        return migrate_v1(raw.get("nodes", []), raw.get("edges", []), raw.get("execution_order"))
    raise ValueError(f"Unbekannte Graph-Version: {version}")
