"""build_model_from_graph — runtime nn.Module from IR."""

from __future__ import annotations

import sys
from pathlib import Path

_CANVAS_DIR = str(Path(__file__).resolve().parent)
if _CANVAS_DIR not in sys.path:
    sys.path.insert(0, _CANVAS_DIR)

from executor import DynamicGraphModule  # noqa: E402
from ir import CanvasGraphIR  # noqa: E402
from shape_propagate import validate_ir_shapes  # noqa: E402


def build_model_from_graph(ir: CanvasGraphIR) -> DynamicGraphModule:
    validate_ir_shapes(ir)
    return DynamicGraphModule(ir)
