"""Backend shape validation for Canvas IR."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ir import CanvasGraphIR, IREdge, IRNode


class ShapeValidationError(Exception):
    def __init__(self, message: str, source_id: str = "", target_id: str = ""):
        super().__init__(message)
        self.source_id = source_id
        self.target_id = target_id


LAYER_SHAPE = {
    "input": ("Variable", "Variable", True),
    "output_node": ("Variable", "Variable", True),
    "dense": ("BD", "BD", False),
    "conv2d": ("BHWC", "BHWC", False),
    "lstm": ("BTC", "BTC", False),
    "embedding": ("BT", "BTC", False),
    "attention": ("BTC", "BTC", False),
    "transformer_block": ("BTC", "BTC", False),
    "layernorm": ("BD", "BD", False),
    "batchnorm": ("BD", "BD", False),
    "add_node": ("BD", "BD", True),
    "merge": ("BD", "BD", True),
    "matmul": ("BD", "BD", True),
}


def _compatible(out_shape: str, in_shape: str) -> bool:
    if "Variable" in (out_shape, in_shape):
        return True
    if out_shape == in_shape:
        return True
    if out_shape == "BHWC" and in_shape == "BD":
        return True
    if out_shape == "BTC" and in_shape == "BD":
        return True
    return False


def validate_ir_shapes(ir: CanvasGraphIR) -> None:
    node_map = ir.node_by_id()
    errors: List[str] = []

    for edge in ir.edges:
        src = node_map.get(edge.source)
        tgt = node_map.get(edge.target)
        if not src or not tgt:
            continue
        src_meta = LAYER_SHAPE.get(src.type)
        tgt_meta = LAYER_SHAPE.get(tgt.type)
        if not src_meta or not tgt_meta:
            continue
        src_out, tgt_in, src_flex, tgt_flex = src_meta[1], tgt_meta[0], src_meta[2], tgt_meta[2]
        if src_flex and tgt_flex:
            continue
        if not _compatible(src_out, tgt_in):
            errors.append(
                f"Shape mismatch: {src.type} ({src.id}) outputs {src_out} "
                f"→ {tgt.type} ({tgt.id}) expects {tgt_in}"
            )

    if errors:
        raise ShapeValidationError("\n".join(errors))
