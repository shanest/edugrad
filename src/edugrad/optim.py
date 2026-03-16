"""Optimizers update model parameters using computed gradients."""

from typing import Iterable

import numpy as np
from edugrad.tensor import Tensor


class Optimizer:

    def __init__(self, params: Iterable[Tensor]):
        self._cur_step = 0
        self.params = params

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros(param.grad.shape)


class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-4):
        super(SGD, self).__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.value -= self.lr * param.grad
        self._cur_step += 1
