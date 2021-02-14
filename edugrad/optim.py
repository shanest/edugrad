""" Copyright 2020-2021 Shane Steinert-Threlkeld

    This file is part of edugrad.

    edugrad is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    edugrad is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with edugrad.  If not, see <https://www.gnu.org/licenses/>.
"""

""" An optimizer updates some parameters. """

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
    def __init__(self, params: Iterable[Tensor], lr=1e-4):
        super(SGD, self).__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.value -= self.lr * param.grad
        self._cur_step += 1
