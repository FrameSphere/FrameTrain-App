"""Tensor mit automatischer Ableitung (Backpropagation)."""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Hebt NumPy-Broadcasting im Gradienten wieder auf."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad: bool = False, parents: Iterable["Tensor"] = (),
                 back: Optional[Callable[[np.ndarray], None]] = None):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)
        self.grad: Optional[np.ndarray] = None
        self._parents = tuple(parents)
        self._back = back

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def backward(self) -> None:
        order, seen = [], set()

        def visit(t: "Tensor") -> None:
            if id(t) not in seen:
                seen.add(id(t))
                for p in t._parents:
                    visit(p)
                order.append(t)

        visit(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            if t._back is not None and t.grad is not None:
                t._back(t.grad)

    def _collect(self, grad: np.ndarray) -> None:
        self.grad = grad if self.grad is None else self.grad + grad

    def __add__(self, other):
        o = other if isinstance(other, Tensor) else Tensor(other)

        def back(g):
            self._collect(_unbroadcast(g, self.shape))
            o._collect(_unbroadcast(g, o.shape))
        return Tensor(self.data + o.data, parents=(self, o), back=back)

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else Tensor(-np.asarray(other, dtype=float)))

    def __mul__(self, other):
        o = other if isinstance(other, Tensor) else Tensor(other)

        def back(g):
            self._collect(_unbroadcast(g * o.data, self.shape))
            o._collect(_unbroadcast(g * self.data, o.shape))
        return Tensor(self.data * o.data, parents=(self, o), back=back)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = other if isinstance(other, Tensor) else Tensor(other)
        return self * o.pow(-1.0)

    def pow(self, p: float):
        return Tensor(self.data ** p, parents=(self,),
                      back=lambda g: self._collect(g * p * self.data ** (p - 1)))

    def __matmul__(self, other):
        o = other

        def back(g):
            self._collect(_unbroadcast(g @ np.swapaxes(o.data, -1, -2), self.shape))
            o._collect(_unbroadcast(np.swapaxes(self.data, -1, -2) @ g, o.shape))
        return Tensor(self.data @ o.data, parents=(self, o), back=back)

    def sum(self, axis=None, keepdims: bool = False):
        def back(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._collect(np.broadcast_to(g, self.shape).copy())
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), parents=(self,), back=back)

    def mean(self, axis=None, keepdims: bool = False):
        count = self.data.size if axis is None else np.prod(
            [self.shape[a] for a in (axis if isinstance(axis, tuple) else (axis,))])
        return self.sum(axis, keepdims) * (1.0 / count)

    def exp(self):
        out = np.exp(self.data)
        return Tensor(out, parents=(self,), back=lambda g: self._collect(g * out))

    def log(self):
        return Tensor(np.log(self.data), parents=(self,), back=lambda g: self._collect(g / self.data))

    def tanh(self):
        out = np.tanh(self.data)
        return Tensor(out, parents=(self,), back=lambda g: self._collect(g * (1 - out ** 2)))

    def relu(self):
        return Tensor(np.maximum(self.data, 0), parents=(self,), back=lambda g: self._collect(g * (self.data > 0)))

    def sigmoid(self):
        out = 1 / (1 + np.exp(-self.data))
        return Tensor(out, parents=(self,), back=lambda g: self._collect(g * out * (1 - out)))

    def gelu(self):
        from math import erf, pi, sqrt
        cdf = 0.5 * (1 + np.vectorize(lambda v: erf(v / sqrt(2)))(self.data))
        pdf = np.exp(-self.data ** 2 / 2) / sqrt(2 * pi)
        return Tensor(self.data * cdf, parents=(self,), back=lambda g: self._collect(g * (cdf + self.data * pdf)))

    def softmax(self, axis: int = -1):
        return self.log_softmax(axis).exp()

    def log_softmax(self, axis: int = -1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        out = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        probs = np.exp(out)

        def back(g):
            self._collect(g - probs * g.sum(axis=axis, keepdims=True))
        return Tensor(out, parents=(self,), back=back)

    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape), parents=(self,),
                      back=lambda g: self._collect(g.reshape(self.shape)))

    def transpose(self, a: int, b: int):
        return Tensor(np.swapaxes(self.data, a, b), parents=(self,),
                      back=lambda g: self._collect(np.swapaxes(g, a, b)))

    def __getitem__(self, index):
        def back(g):
            full = np.zeros_like(self.data)
            np.add.at(full, index, g)
            self._collect(full)
        return Tensor(self.data[index], parents=(self,), back=back)
