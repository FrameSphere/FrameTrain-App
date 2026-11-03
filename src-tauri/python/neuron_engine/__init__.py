"""NeuronEngine: ein kleines Deep-Learning-Framework nur auf NumPy.

Jede Rechnung baut einen Graphen, backward() geht ihn rueckwaerts durch. Kein
PyTorch, kein GPU-Beschleuniger: gedacht zum Verstehen und fuer kleine Modelle.
"""

from .tensor import Tensor
from .nn import (Attention, Conv2d, Dropout, Embedding, GELU, LayerNorm, Linear, Module, ReLU, Sequential, Sigmoid,
                 Softmax, Tanh)
from .train import Adam, SGD, cross_entropy, mse

__all__ = ["Tensor", "Module", "Sequential", "Linear", "Conv2d", "Embedding", "Attention", "LayerNorm", "Dropout",
           "ReLU", "GELU", "Sigmoid", "Tanh", "Softmax", "SGD", "Adam", "cross_entropy", "mse"]
