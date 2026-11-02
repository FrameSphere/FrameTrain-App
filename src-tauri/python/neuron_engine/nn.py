"""Neuronen als Schichten: Gewichte w, Bias b, Aktivierung f."""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional

import numpy as np

from .tensor import Tensor


class Module:
    training = True

    def parameters(self) -> Iterator[Tensor]:
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Module):
                        yield from item.parameters()

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for value in self.__dict__.values():
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, Module):
                    item.train(mode)
        return self

    def __call__(self, *args):
        return self.forward(*args)


class Sequential(Module):
    def __init__(self, layers: List[Module]):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _uniform(rng, bound: float, shape) -> Tensor:
    return Tensor(rng.uniform(-bound, bound, shape), requires_grad=True)


class Linear(Module):
    """y = x @ w + b"""

    def __init__(self, n_in: int, n_out: int, bias: bool = True, rng=None):
        rng = rng or np.random.default_rng()
        self.w = _uniform(rng, 1 / math.sqrt(n_in), (n_out, n_in))
        self.b = _uniform(rng, 1 / math.sqrt(n_in), (n_out,)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.w.transpose(-1, -2)
        return out + self.b if self.b is not None else out


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class GELU(Module):
    def forward(self, x):
        return x.gelu()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Softmax(Module):
    def forward(self, x):
        return x.softmax()


class Dropout(Module):
    def __init__(self, p: float = 0.1, rng=None):
        self.p, self.rng = p, rng or np.random.default_rng()

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0:
            return x
        return x * ((self.rng.random(x.shape) >= self.p) / (1 - self.p))


class Embedding(Module):
    def __init__(self, vocab: int, dim: int, rng=None):
        rng = rng or np.random.default_rng()
        self.w = Tensor(rng.normal(0, 1, (vocab, dim)), requires_grad=True)

    def forward(self, ids) -> Tensor:
        return self.w[np.asarray(ids.data if isinstance(ids, Tensor) else ids, dtype=np.int64)]


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.w = Tensor(np.ones(dim), requires_grad=True)
        self.b = Tensor(np.zeros(dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(-1, keepdims=True)
        return centered * (var + self.eps).pow(-0.5) * self.w + self.b


class Conv2d(Module):
    """Faltung als Multiplikation ueber alle Fensterpositionen."""

    def __init__(self, n_in: int, n_out: int, kernel: int, stride: int = 1, padding: int = 0, rng=None):
        rng = rng or np.random.default_rng()
        bound = 1 / math.sqrt(n_in * kernel * kernel)
        self.w = _uniform(rng, bound, (n_out, n_in, kernel, kernel))
        self.b = _uniform(rng, bound, (n_out,))
        self.kernel, self.stride, self.padding = kernel, stride, padding

    def forward(self, x: Tensor) -> Tensor:
        k, s, pad = self.kernel, self.stride, self.padding
        n, c, h, w = x.shape
        H, W = (h + 2 * pad - k) // s + 1, (w + 2 * pad - k) // s + 1
        idx = np.arange((h + 2 * pad) * (w + 2 * pad)).reshape(h + 2 * pad, w + 2 * pad)
        windows = np.lib.stride_tricks.sliding_window_view(idx, (k, k))[::s, ::s].reshape(H * W, k * k)

        def unfold(a: np.ndarray) -> np.ndarray:
            padded = np.pad(a, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
            return padded.reshape(n, c, -1)[:, :, windows]  # [N, C, Fenster, k*k]

        def back(g):
            full = np.zeros((n, c, (h + 2 * pad) * (w + 2 * pad)))
            np.add.at(full, (slice(None), slice(None), windows), g)
            full = full.reshape(n, c, h + 2 * pad, w + 2 * pad)
            x._collect(full[:, :, pad:pad + h, pad:pad + w])

        patches = Tensor(unfold(x.data), parents=(x,), back=back)  # [N, C, P, K]
        flat = self.w.reshape(self.w.shape[0], c, k * k)
        out = patches.transpose(1, 2).reshape(n, H * W, c * k * k) @ flat.reshape(flat.shape[0], c * k * k).transpose(0, 1)
        return out.transpose(1, 2).reshape(n, self.w.shape[0], H, W) + self.b.reshape(1, -1, 1, 1)


class Attention(Module):
    """Mehrkoepfige Selbst-Aufmerksamkeit [B, T, D]."""

    def __init__(self, dim: int, heads: int, rng=None):
        rng = rng or np.random.default_rng()
        self.heads, self.dim = heads, dim
        self.q = Linear(dim, dim, rng=rng)
        self.k = Linear(dim, dim, rng=rng)
        self.v = Linear(dim, dim, rng=rng)
        self.out = Linear(dim, dim, rng=rng)

    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        b, t, d = x.shape
        size = d // self.heads

        def split(v: Tensor) -> Tensor:
            return v.reshape(b, t, self.heads, size).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = (q @ k.transpose(-1, -2)) * (1 / math.sqrt(size))
        if causal:
            scores = scores + np.triu(np.full((t, t), -1e9), 1)
        mixed = scores.softmax(-1) @ v
        return self.out(mixed.transpose(1, 2).reshape(b, t, d))
