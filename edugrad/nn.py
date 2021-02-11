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

"""Defines the Module class, basic building block of neural networks.

Also includes a Linear Module, for computing xW + b.
"""

from typing import Callable, List
import numpy as np

from edugrad.tensor import Tensor
import edugrad.ops as ops


class Module:
    """Modules are components of computation graphs.

    They contain parameters (Tensors) and/or sub-modules.

    Most importantly, a Module has a `forward` method, which implements the
    forward pass of a computation graph.  This forward pass should use `ops`
    which ensures that the computation graph gets built dynamically, so that
    `backward` can be used to compute gradients.

    Attributes:
        _params: dictionary of Tensors, trainable Tensor parameters
        _modules: sub-modules of this module
    """
    def __init__(self):
        self._params = dict()
        self._modules = dict()

    def zero_grad(self) -> None:
        """Set gradients of all parameters to zero. """
        for param in self.parameters():
            param.grad = np.zeros(param.value.shape)

    def parameters(self) -> List[Tensor]:
        """Return a list of all parameters of this module. """
        params = list(self._params.values())
        for module in self._modules:
            params.extend(self._modules[module].parameters())
        return params

    def forward(self, *inputs: List[Tensor]) -> Tensor:
        raise NotImplementedError

    def __setattr__(self, key, value) -> None:
        """Register modules and params when assigned. """
        if isinstance(value, Tensor):
            self._params[key] = value
        elif isinstance(value, Module):
            self._modules[key] = value
        super(Module, self).__setattr__(key, value)

    def __call__(self, *inputs: List[Tensor]) -> Tensor:
        return self.forward(*inputs)


class Linear(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        initializer: Callable = np.random.random,
        bias: bool = True,
    ):
        """A Linear module computes defines weights W, optionally biases b, and
        computers wX + b.

        Weight vector will have shape (input size, output size)

        Args:
            input_size: dimension of input vectors
            output_size: dimension of output vectors
            initializer: how to initialize weights and biases
            bias: whether or not to include the bias term; not needed for, e.g. embeddings
        """
        super(Linear, self).__init__()
        self.weights = Tensor(initializer((input_size, output_size)), name="W")
        self.bias = bias
        if self.bias:
            self.biases = Tensor(initializer((1, output_size)), name="b")

    def forward(self, inputs: Tensor):
        mul_node = ops.matmul(inputs, self.weights)
        if self.bias:
            # NOTE: this is a hack-ish way of handling shape issues with biases
            expanded_biases = ops.copy_rows(self.biases, num=inputs.value.shape[0])
            return ops.add(mul_node, expanded_biases)
        return mul_node
