"""Create nn.Module instances from IR node types."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn


def _int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return default


LAYER_TYPES = frozenset({
    "dense", "conv2d", "embedding", "lstm", "attention", "transformer_block",
    "layernorm", "batchnorm", "dropout",
    "relu", "gelu", "sigmoid", "softmax", "tanh", "leaky_relu", "silu",
})

OP_TYPES = frozenset({
    "add_node", "multiply_node", "matmul", "normalize", "reshape", "transpose",
    "merge", "split_node", "pool",
})

SKIP_TYPES = frozenset({"input", "output_node", "optimizer", "loss", "scheduler"})


def is_layer_node(node_type: str) -> bool:
    return node_type in LAYER_TYPES


def is_op_node(node_type: str) -> bool:
    return node_type in OP_TYPES


def create_layer(node_type: str, params: Dict[str, Any]) -> Optional[nn.Module]:
    p = params or {}
    if node_type == "dense":
        return nn.Linear(
            _int(p.get("inputSize"), 128),
            _int(p.get("outputSize"), 256),
            bias=_bool(p.get("bias"), True),
        )
    if node_type == "conv2d":
        pad = p.get("padding", 1)
        if pad == "same":
            padding = "same"
        else:
            padding = _int(pad, 1)
        return nn.Conv2d(
            _int(p.get("inChannels"), 3),
            _int(p.get("outChannels"), 64),
            kernel_size=_int(p.get("kernelSize"), 3),
            stride=_int(p.get("stride"), 1),
            padding=padding,
            groups=_int(p.get("groups"), 1),
        )
    if node_type == "embedding":
        return nn.Embedding(
            _int(p.get("vocabSize"), 50000),
            _int(p.get("embeddingDim"), 512),
            padding_idx=_int(p.get("paddingIdx"), 0),
        )
    if node_type == "lstm":
        num_layers = _int(p.get("numLayers"), 2)
        drop = _float(p.get("dropout"), 0.1) if num_layers > 1 else 0.0
        return nn.LSTM(
            _int(p.get("inputSize"), 256),
            _int(p.get("hiddenSize"), 512),
            num_layers=num_layers,
            bidirectional=_bool(p.get("bidirectional")),
            dropout=drop,
            batch_first=True,
        )
    if node_type == "attention":
        return nn.MultiheadAttention(
            _int(p.get("embedDim"), 512),
            _int(p.get("numHeads"), 8),
            dropout=_float(p.get("dropout"), 0.1),
            batch_first=True,
        )
    if node_type == "transformer_block":
        return nn.TransformerEncoderLayer(
            d_model=_int(p.get("embedDim"), 512),
            nhead=_int(p.get("numHeads"), 8),
            dim_feedforward=_int(p.get("ffnDim"), 2048),
            dropout=_float(p.get("dropout"), 0.1),
            batch_first=True,
            norm_first=True,
        )
    if node_type == "layernorm":
        return nn.LayerNorm(
            _int(p.get("normalizedShape"), 512),
            eps=_float(p.get("eps"), 1e-5),
            elementwise_affine=_bool(p.get("affine"), True),
        )
    if node_type == "batchnorm":
        return nn.BatchNorm1d(
            _int(p.get("numFeatures"), 64),
            eps=_float(p.get("eps"), 1e-5),
            momentum=_float(p.get("momentum"), 0.1),
            affine=_bool(p.get("affine"), True),
        )
    if node_type == "dropout":
        return nn.Dropout(p=_float(p.get("p"), 0.1), inplace=_bool(p.get("inplace")))
    if node_type == "relu":
        return nn.ReLU(inplace=_bool(p.get("inplace")))
    if node_type == "gelu":
        return nn.GELU(approximate=str(p.get("approximate", "none")))
    if node_type == "sigmoid":
        return nn.Sigmoid()
    if node_type == "softmax":
        return nn.Softmax(dim=_int(p.get("dim"), -1))
    if node_type == "tanh":
        return nn.Tanh()
    if node_type == "leaky_relu":
        return nn.LeakyReLU(negative_slope=_float(p.get("negativeSlope"), 0.01))
    if node_type == "silu":
        return nn.SiLU()
    return None
