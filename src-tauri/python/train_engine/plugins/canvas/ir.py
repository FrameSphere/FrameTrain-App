"""Canvas Graph IR — parsed from config.canvas_graph JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IRNode:
    id: str
    type: str
    category: str
    params: Dict[str, Any] = field(default_factory=dict)
    position: Optional[Dict[str, float]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IRNode":
        return cls(
            id=str(d["id"]),
            type=str(d.get("type", "unknown")),
            category=str(d.get("category", "")),
            params=dict(d.get("params") or {}),
            position=d.get("position"),
        )


@dataclass
class IREdge:
    source: str
    target: str
    id: Optional[str] = None
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IREdge":
        return cls(
            id=d.get("id"),
            source=str(d["source"]),
            target=str(d["target"]),
            source_handle=d.get("sourceHandle") or d.get("source_handle"),
            target_handle=d.get("targetHandle") or d.get("target_handle"),
        )


@dataclass
class IRTrainingSpec:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    clip_grad: float = 1.0
    loss: str = "cross_entropy"
    loss_reduction: str = "mean"
    label_smoothing: float = 0.0
    scheduler: str = "cosine"
    warmup_steps: int = 0
    min_lr: float = 0.0
    num_classes: int = 10
    task_type: str = "classification"
    precision: str = "fp32"
    grad_accum: int = 1
    gpu: str = "cpu"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IRTrainingSpec":
        return cls(
            epochs=int(d.get("epochs", 10)),
            batch_size=int(d.get("batchSize", d.get("batch_size", 32))),
            learning_rate=float(d.get("learningRate", d.get("learning_rate", 0.001))),
            weight_decay=float(d.get("weightDecay", d.get("weight_decay", 0.01))),
            optimizer=str(d.get("optimizer", "adamw")),
            clip_grad=float(d.get("clipGrad", d.get("clip_grad", 1.0))),
            loss=str(d.get("loss", "cross_entropy")),
            loss_reduction=str(d.get("lossReduction", d.get("loss_reduction", "mean"))),
            label_smoothing=float(d.get("labelSmoothing", d.get("label_smoothing", 0.0))),
            scheduler=str(d.get("scheduler", "cosine")),
            warmup_steps=int(d.get("warmupSteps", d.get("warmup_steps", 0))),
            min_lr=float(d.get("minLr", d.get("min_lr", 0.0))),
            num_classes=int(d.get("numClasses", d.get("num_classes", 10))),
            task_type=str(d.get("taskType", d.get("task_type", "classification"))),
            precision=str(d.get("precision", "fp32")),
            grad_accum=int(d.get("gradAccum", d.get("grad_accum", 1))),
            gpu=str(d.get("gpu", "cpu")),
        )


@dataclass
class IRDataSpec:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IRDataSpec":
        return cls(type=str(d.get("type", "default")), params=dict(d.get("params") or {}))


@dataclass
class CanvasGraphIR:
    version: int
    nodes: List[IRNode]
    edges: List[IREdge]
    execution_order: List[str]
    training: IRTrainingSpec
    data: Optional[IRDataSpec] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def node_by_id(self) -> Dict[str, IRNode]:
        return {n.id: n for n in self.nodes}

    def in_edges_map(self) -> Dict[str, List[str]]:
        m: Dict[str, List[str]] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            if e.target in m:
                m[e.target].append(e.source)
        return m


def parse_ir(raw: Any) -> CanvasGraphIR:
    if not raw or not isinstance(raw, dict):
        raise ValueError("canvas_graph ist leer oder kein Objekt")
    if raw.get("version") != 1:
        raise ValueError(f"Unsupported IR version: {raw.get('version')}")
    nodes = [IRNode.from_dict(n) for n in raw.get("nodes", [])]
    edges = [IREdge.from_dict(e) for e in raw.get("edges", [])]
    order = list(raw.get("execution_order", []))
    if not order and nodes:
        order = [n.id for n in nodes]
    training = IRTrainingSpec.from_dict(raw.get("training") or {})
    data_raw = raw.get("data")
    data = IRDataSpec.from_dict(data_raw) if data_raw else None
    return CanvasGraphIR(
        version=1,
        nodes=nodes,
        edges=edges,
        execution_order=order,
        training=training,
        data=data,
        metadata=dict(raw.get("metadata") or {}),
    )


def is_non_empty_ir(raw: Any) -> bool:
    try:
        ir = parse_ir(raw)
        return len(ir.nodes) > 0 and len(ir.execution_order) > 0
    except Exception:
        return False
