"""Sprache der Node-Definitionen: Port-Typen, Parameter und Registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

# Formen je Port-Typ. "Any" nimmt alles an und wird nicht geprueft.
PORT_TYPES: Dict[str, str] = {
    "Tensor": "beliebiger Tensor",
    "Dataset": "Datenstrom",
    "Scalar": "einzelner Wert",
    "Any": "beliebig",
}

AUTO = "auto"


@dataclass(frozen=True)
class Port:
    id: str
    type: str = "Tensor"
    label: str = ""
    # Optionale Eingaenge fallen auf einen anderen Port zurueck (z. B. key -> query).
    fallback: Optional[str] = None


@dataclass(frozen=True)
class Param:
    key: str
    default: Any = None
    # planned: im Editor sichtbar, aber noch ohne Wirkung. Bleibt im Katalog
    # markiert, damit die Oberflaeche es sagen kann, statt es zu verschweigen.
    planned: bool = False


@dataclass
class NodeSpec:
    type: str
    family: str
    inputs: List[Port] = field(default_factory=list)
    outputs: List[Port] = field(default_factory=list)
    params: List[Param] = field(default_factory=list)
    # build(params, input_shapes) -> nn.Module, None bei Ops ohne Gewichte
    build: Optional[Callable[..., Any]] = None
    # run(module, params, inputs) -> {port: tensor}
    run: Optional[Callable[..., Dict[str, Any]]] = None
    # Nur Struktur fuer Daten und Training: laeuft nicht im forward.
    structural: bool = False


REGISTRY: Dict[str, NodeSpec] = {}


def ports(*items: Any) -> List[Port]:
    return [p if isinstance(p, Port) else Port(p) for p in items]


def params(**items: Any) -> List[Param]:
    return [v if isinstance(v, Param) else Param(k, v) for k, v in items.items()]


def node(name: str, family: str, /, inputs: Sequence[Any] = ("in",), outputs: Sequence[Any] = ("out",),
         build: Optional[Callable[..., Any]] = None, run: Optional[Callable[..., Dict[str, Any]]] = None,
         structural: bool = False, **param_defaults: Any) -> NodeSpec:
    spec = NodeSpec(type=name, family=family, inputs=ports(*inputs), outputs=ports(*outputs),
                    params=[v if isinstance(v, Param) else Param(k, v) for k, v in param_defaults.items()],
                    build=build, run=run, structural=structural)
    if name in REGISTRY:
        raise ValueError(f"Node doppelt definiert: {name}")
    REGISTRY[name] = spec
    return spec


def catalog() -> Dict[str, Any]:
    """Maschinenlesbarer Katalog fuer den Editor."""
    from . import nodes  # noqa: F401  (fuellt REGISTRY)
    return {
        "version": 1,
        "nodes": [
            {
                "type": s.type, "family": s.family, "structural": s.structural,
                "inputs": [{"id": p.id, "type": p.type, "optional": p.fallback is not None} for p in s.inputs],
                "outputs": [{"id": p.id, "type": p.type} for p in s.outputs],
                "params": [{"key": p.key, "default": p.default, "planned": p.planned} for p in s.params],
            }
            for s in sorted(REGISTRY.values(), key=lambda s: (s.family, s.type))
        ],
    }
