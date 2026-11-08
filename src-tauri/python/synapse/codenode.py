"""Eigener Code-Node: Ports und Einstellungen stehen im Code selbst.

    class Node(nn.Module):
        outputs = ("mu", "logvar")                 # Ausgaenge, sonst einer: "out"

        def __init__(self, shape, hidden: int = 128):   # hidden -> Einstellung
            ...

        def forward(self, x, skip=None):           # x, skip -> Eingaenge, skip optional
            ...

Der erste Eingang heisst immer "in" (wie bei allen anderen Nodes), weitere wie
ihr Argument. Mit einem Eingang ist shape die Form davor, mit mehreren ein Dict
je verbundenem Eingang, nach Argument-Namen.
"""

from __future__ import annotations

import inspect
import keyword
import typing
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spec import Port

# Diese Schluessel nutzt der Editor selbst in den Node-Parametern.
RESERVED = {"code", "label", "icon", "color", "description", "libraryId", "libraryVersion"}

DEFAULT_CODE = """class Node(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.scale
"""

_EMPTY = inspect.Parameter.empty
_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


@dataclass(frozen=True)
class CodeInfo:
    cls: Any
    inputs: Tuple[Port, ...]
    args: Tuple[str, ...]  # forward-Argument je Eingang, gleiche Reihenfolge
    outputs: Tuple[Port, ...]
    params: Tuple[Dict[str, Any], ...]

    @property
    def single(self) -> bool:
        return len(self.inputs) == 1


def compile_code(source: str) -> Any:
    """Der Text definiert eine Klasse Node(nn.Module).

    Laeuft lokal im selben Prozess wie Dev Train, ohne Sandbox.
    """
    scope: Dict[str, Any] = {"torch": torch, "nn": nn, "F": F, "Literal": typing.Literal,
                             "Optional": typing.Optional}
    try:
        code = compile(source, "<code-node>", "exec", dont_inherit=True)
    except SyntaxError as e:
        raise ValueError(f"Zeile {e.lineno}: {e.msg}") from None
    exec(code, scope)
    cls = scope.get("Node")
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise ValueError("Der Code muss eine Klasse Node(nn.Module) definieren")
    return cls


def _kind(annotation: Any, default: Any) -> Tuple[str, List[Any]]:
    if typing.get_origin(annotation) is typing.Literal:
        return "select", list(typing.get_args(annotation))
    t = annotation if annotation in (bool, int, float, str) else type(default)
    return {bool: "boolean", int: "int", float: "float"}.get(t, "string"), []


@lru_cache(maxsize=128)
def code_info(source: str) -> CodeInfo:
    cls = compile_code(source)

    fwd = list(inspect.signature(cls.forward).parameters.values())[1:]
    if any(p.kind in _VARIADIC for p in fwd):
        raise ValueError("forward darf kein *args oder **kwargs haben, jeder Eingang braucht einen Namen")
    if not fwd:
        raise ValueError("forward braucht mindestens einen Eingang, z. B. forward(self, x)")
    if fwd[0].default is not _EMPTY:
        raise ValueError(f"Der erste Eingang '{fwd[0].name}' darf keinen Standardwert haben")
    inputs = tuple(Port("in" if i == 0 else p.name, label=p.name, optional=p.default is not _EMPTY)
                   for i, p in enumerate(fwd))

    names = getattr(cls, "outputs", ("out",))
    if isinstance(names, str):
        names = (names,)
    names = tuple(names)
    if not names or any(not isinstance(n, str) or not n.isidentifier() or keyword.iskeyword(n) for n in names):
        raise ValueError("outputs muss eine Liste von Namen sein, z. B. outputs = (\"mu\", \"logvar\")")
    if len(set(names)) != len(names):
        raise ValueError("outputs enthaelt einen Namen doppelt")

    init = list(inspect.signature(cls.__init__).parameters.values())[1:]
    if not init or init[0].name != "shape":
        raise ValueError("__init__ braucht als ersten Parameter shape, z. B. __init__(self, shape)")
    params = []
    for p in init[1:]:
        if p.kind in _VARIADIC:
            continue
        if p.default is _EMPTY:
            raise ValueError(f"Die Einstellung '{p.name}' braucht einen Standardwert, z. B. {p.name}=128")
        if p.name in RESERVED:
            raise ValueError(f"Der Name '{p.name}' ist fuer den Editor reserviert, bitte anders nennen")
        kind, options = _kind(p.annotation, p.default)
        params.append({"key": p.name, "default": p.default, "kind": kind, "options": options})
    return CodeInfo(cls, inputs, tuple(p.name for p in fwd), tuple(Port(n) for n in names), tuple(params))


def info_of(params: Dict[str, Any]) -> CodeInfo:
    return code_info(str(params.get("code") or DEFAULT_CODE))


def ports_of(params: Dict[str, Any]) -> Tuple[Tuple[Port, ...], Tuple[Port, ...]]:
    """Ports fuer Kanten; kaputter Code faellt auf in/out zurueck, der Fehler kommt beim Bauen."""
    try:
        info = info_of(params)
        return info.inputs, info.outputs
    except Exception:
        return (Port("in"),), (Port("out"),)


def _coerce(value: Any, param: Dict[str, Any]) -> Any:
    kind, default = param["kind"], param["default"]
    if value is None or value == "":
        return default
    if kind == "boolean":
        return value if isinstance(value, bool) else str(value).strip().lower() in ("true", "1", "yes", "ja")
    if kind == "int":
        return int(float(value))
    if kind == "float":
        return float(value)
    if kind == "select":
        return next((o for o in param["options"] if str(o) == str(value)), default)
    return str(value)


def code_kwargs(params: Dict[str, Any], info: CodeInfo) -> Dict[str, Any]:
    out = {}
    for p in info.params:
        try:
            out[p["key"]] = _coerce(params.get(p["key"]), p)
        except (TypeError, ValueError):
            raise ValueError(f"Einstellung '{p['key']}': {params.get(p['key'])!r} ist kein gueltiger Wert") from None
    return out


def code_shape(info: CodeInfo, shapes: Dict[str, Any]) -> Any:
    if info.single:
        return shapes.get("in")
    return {arg: shapes[port.id] for port, arg in zip(info.inputs, info.args) if port.id in shapes}


def build_code(params: Dict[str, Any], shapes: Dict[str, Any]) -> nn.Module:
    info = info_of(params)
    return info.cls(code_shape(info, shapes or {}), **code_kwargs(params, info))


def split_result(result: Any, info: CodeInfo) -> Dict[str, Any]:
    names = [p.id for p in info.outputs]
    if isinstance(result, dict):
        missing = [n for n in names if n not in result]
        if missing:
            raise ValueError(f"forward liefert keinen Wert fuer {', '.join(missing)}")
        return {n: result[n] for n in names}
    if len(names) == 1:
        return {names[0]: result}
    if isinstance(result, (tuple, list)) and len(result) == len(names):
        return dict(zip(names, result))
    raise ValueError(f"forward muss {len(names)} Werte liefern ({', '.join(names)}), als Dict oder Tupel")


def run_code(module: nn.Module, params: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    info = info_of(params)
    kwargs = {arg: inputs[port.id] for port, arg in zip(info.inputs, info.args) if port.id in inputs}
    return split_result(module(**kwargs), info)


def describe(source: str) -> Dict[str, Any]:
    """Fuer die Werkstatt: was der Code als Node bedeutet, oder der Fehler."""
    try:
        info = code_info(source)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "inputs": [{"id": p.id, "label": p.label, "optional": p.optional} for p in info.inputs],
        "outputs": [{"id": p.id} for p in info.outputs],
        "params": [{**p, "options": [str(o) for o in p["options"]]} for p in info.params],
    }


def _code_line(err: BaseException) -> str:
    import traceback
    lines = [f.lineno for f in traceback.extract_tb(err.__traceback__) if f.filename == "<code-node>"]
    return f"Zeile {lines[-1]}: " if lines else ""


def probe(source: str, values: Dict[str, Any], shape: Any) -> Dict[str, Any]:
    """describe plus ein Probelauf: alle Pflicht-Eingaenge bekommen die Beispielform."""
    out = describe(source)
    if not out["ok"]:
        return out
    params = {**(values or {}), "code": source}
    info = code_info(source)
    dims = tuple(int(v) for v in shape) if shape else (1, 16)
    shapes = {p.id: dims for p in info.inputs if not p.optional}
    try:
        mod = build_code(params, shapes)
        with torch.no_grad():
            result = run_code(mod.eval(), params, {k: torch.zeros(*v) for k, v in shapes.items()})
        out["probe"] = {k: list(v.shape) for k, v in result.items()}
        out["parameters"] = sum(p.numel() for p in mod.parameters())
    except Exception as e:
        out["probeError"] = f"{_code_line(e)}{type(e).__name__}: {e}"
    return out
