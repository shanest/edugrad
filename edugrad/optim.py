""" An optimizer updates some parameters. """

from typing import Iterable

from .tensor import Tensor


class Optimizer:
    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr=1e-4):
        self.lr = lr
        self._cur_step = 0
        self.params = params

    def step(self):
        for param in self.params:
            param.value -= self.lr * param.grad
        self._cur_step += 1
