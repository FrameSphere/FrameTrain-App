"""Formen-Pruefung: fuehrt den Graphen auf Meta-Tensoren aus, ohne Gewichte im RAM."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .graph import Graph, load_graph
from .runtime import SynapseModule


def check_graph(raw: Any, input_shape: Tuple[int, ...], input_dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
    """Liefert je Node Ausgabeformen oder den Fehler, dazu die Parameterzahl."""
    graph: Graph = load_graph(raw) if isinstance(raw, dict) else raw
    result: Dict[str, Any] = {}
    errors = []

    def seen(nid: str, shapes: Optional[Dict[str, Tuple[int, ...]]], err: Optional[Exception]) -> None:
        if err is not None:
            errors.append({"node": nid, "message": str(err)})
        else:
            result[nid] = {"outputs": {k: list(v) for k, v in (shapes or {}).items()}}

    with torch.device("meta"):
        mod = SynapseModule(graph, input_shape=input_shape, input_dtype=input_dtype, on_node=seen)
    params = sum(p.numel() for p in mod.parameters())
    for nid, layer in mod.layers.items():
        if nid in result:
            result[nid]["params"] = sum(p.numel() for p in layer.parameters())
    return {"ok": not errors, "nodes": result, "errors": errors, "params": params}


def _check_gan(raw: Dict[str, Any], shape: Any) -> Dict[str, Any]:
    from types import SimpleNamespace as NS
    from .generative import GanModel
    nodes = [NS(type=n.get("type"), params=n.get("params") or {}) for n in raw.get("nodes", [])]
    latent = next((int(n.params.get("dim", 64)) for n in nodes if n.type == "latent"), 64)
    with torch.device("meta"):
        model = GanModel(nodes, latent, tuple(shape))
    return {"ok": True, "nodes": {}, "errors": [], "params": sum(p.numel() for p in model.parameters()),
            "input_shape": list(shape)}


def guess_input(raw: Dict[str, Any]) -> Tuple[Tuple[int, ...], str]:
    """Eingabeform eines Graphen aus Loader, Input-Node oder erstem Layer."""
    nodes_raw = raw.get("nodes", [])
    by_type = {n.get("type"): n.get("params") or {} for n in reversed(nodes_raw)}
    if "timestep" in by_type:
        p = by_type.get("image_loader") or {}
        size, ch = int(p.get("imageSize", 32)), int(p.get("channels", 3))
        return {"out": (1, ch, size, size), "decoder": (1, 1)}, "float32"
    if "image_loader" in by_type:
        p = by_type["image_loader"]
        size, ch = int(p.get("imageSize", 224)), int(p.get("channels", 3))
        return (1, min(ch, 3) if ch != 1 else 1, size, size), "float32"
    if "audio_loader" in by_type:
        p = by_type["audio_loader"]
        samples = int(float(p.get("seconds", 1.0)) * int(p.get("sampleRate", 16000)))
        if "mel_spectrogram" in by_type:
            m = by_type["mel_spectrogram"]
            return (1, 1, int(m.get("nMels", 64)), samples // int(m.get("hopLength", 160)) + 1), "float32"
        return (1, 1, samples), "float32"
    if "video_loader" in by_type:
        p = by_type["video_loader"]
        return (1, 3, int(p.get("numFrames", 8)), int(p.get("imageSize", 112)), int(p.get("imageSize", 112))), "float32"
    if "text_loader" in by_type:
        t = by_type.get("tokenizer") or {}
        return (1, int(t.get("maxLength", 512))), "int64"
    if "detection_loader" in by_type:
        size = int(by_type["detection_loader"].get("imageSize", 128))
        return (1, 3, size, size), "float32"
    if "seq2seq_loader" in by_type:
        t = by_type.get("tokenizer") or {}
        n = int(t.get("maxLength", 32))
        return {"out": (1, n), "decoder": (1, n)}, "int64"
    if "mask_loader" in by_type:
        size = int(by_type["mask_loader"].get("imageSize", 128))
        return (1, 3, size, size), "float32"
    order = raw.get("execution_order") or []
    layers = sorted((n for n in nodes_raw if n.get("type") in ("dense", "lstm", "conv2d", "embedding")),
                    key=lambda n: order.index(n["id"]) if n["id"] in order else 10 ** 6)
    first = (layers[0].get("params") or {}) if layers and layers[0].get("type") == "dense" else {}
    if isinstance(first.get("inputSize"), (int, float)):
        # Eine feste Dense-Groesse sagt mehr als der Standardwert des Input-Nodes.
        return (1, int(first["inputSize"])), "float32"
    if "input" in by_type:
        dims = [v for v in str(by_type["input"].get("shape", "")).replace(" ", "").split(",") if v]
        try:
            shape = tuple(max(int(v), 1) for v in dims)
        except ValueError:
            shape = ()
        if shape:
            dtype = str(by_type["input"].get("dtype", "float32"))
            return shape, dtype
    for n in sorted(nodes_raw, key=lambda n: (raw.get("execution_order") or []).index(n["id"])
                    if n["id"] in (raw.get("execution_order") or []) else 10 ** 6):
        p, t = n.get("params") or {}, n.get("type")
        known = {k: v for k, v in p.items() if isinstance(v, (int, float))}
        if t == "dense" and "inputSize" in known:
            return (1, int(known["inputSize"])), "float32"
        if t == "conv2d" and "inChannels" in known:
            return (1, int(known["inChannels"]), 32, 32), "float32"
        if t == "lstm" and "inputSize" in known:
            return (1, 8, int(known["inputSize"])), "float32"
        if t == "embedding":
            return (1, 32), "int64"
    return (1, 16), "float32"


def check_raw(raw: Dict[str, Any], input_shape: Any = None, dtype: Optional[str] = None) -> Dict[str, Any]:
    """Pruefung fuer den Dienst: gibt Fehler als Ergebnis zurueck, wirft nicht."""
    shape, guessed = guess_input(raw) if input_shape is None else (tuple(input_shape), "float32")
    first = shape["out"] if isinstance(shape, dict) else shape
    try:
        if any((n.get("params") or {}).get("role") for n in raw.get("nodes", [])):
            return _check_gan(raw, first)
        graph = load_graph(raw)
        result = check_graph(graph, shape, getattr(torch, dtype or guessed))
    except Exception as e:
        return {"ok": False, "nodes": {}, "errors": [{"node": "", "message": str(e)}], "params": 0,
                "input_shape": list(first)}
    result["input_shape"] = list(first)
    result["edge_sources"] = {
        f"{e.target}:{e.target_port}": [e.source, e.source_port] for e in graph.edges}
    return result
