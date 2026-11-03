"""Fehler messen und Gewichte anpassen."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .tensor import Tensor


def cross_entropy(logits: Tensor, targets) -> Tensor:
    y = np.asarray(targets, dtype=np.int64)
    log_p = logits.log_softmax(-1)
    return -log_p[np.arange(len(y)), y].mean()


def mse(pred: Tensor, target) -> Tensor:
    diff = pred - target
    return (diff * diff).mean()


class SGD:
    def __init__(self, params: Iterable[Tensor], lr: float = 0.01, momentum: float = 0.0):
        self.params: List[Tensor] = list(params)
        self.lr, self.momentum = lr, momentum
        self.state = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.state[i] = self.momentum * self.state[i] + p.grad
                p.data -= self.lr * self.state[i]


class Adam(SGD):
    def __init__(self, params: Iterable[Tensor], lr: float = 0.001, betas=(0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, lr)
        self.betas, self.eps, self.t = betas, eps, 0
        self.var = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.state[i] = b1 * self.state[i] + (1 - b1) * p.grad
            self.var[i] = b2 * self.var[i] + (1 - b2) * p.grad ** 2
            mean = self.state[i] / (1 - b1 ** self.t)
            var = self.var[i] / (1 - b2 ** self.t)
            p.data -= self.lr * mean / (np.sqrt(var) + self.eps)
