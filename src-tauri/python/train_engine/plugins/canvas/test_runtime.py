"""Quick validation: build_model_from_graph from IR.

Run from train_engine root:
  python3 -m plugins.canvas.test_runtime
"""

import torch

from plugins.canvas.ir import CanvasGraphIR, IRNode, IREdge, IRTrainingSpec
from plugins.canvas.model_builder import build_model_from_graph


def test_linear_chain():
    ir = CanvasGraphIR(
        version=1,
        nodes=[
            IRNode("in", "input", "data", {}),
            IRNode("d1", "dense", "layer", {"inputSize": 32, "outputSize": 16}),
            IRNode("d2", "dense", "layer", {"inputSize": 16, "outputSize": 4}),
        ],
        edges=[
            IREdge("in", "d1"),
            IREdge("d1", "d2"),
        ],
        execution_order=["in", "d1", "d2"],
        training=IRTrainingSpec(num_classes=4),
    )
    model = build_model_from_graph(ir)
    x = torch.randn(2, 32)
    y = model(x)
    assert y.shape == (2, 4), y.shape
    print("OK linear chain", y.shape)


if __name__ == "__main__":
    test_linear_chain()
