from typing import Callable, List
import numpy as np

from .tensor import Tensor
from . import ops


class Module:
    def __init__(self):
        self._params = dict()
        self._modules = dict()

    def zero_grad(self) -> None:
        """ Set gradients of all parameters to zero. """
        for param in self.parameters():
            param.grad = np.zeros(param.value.shape)

    def parameters(self) -> List[Tensor]:
        """ Return a list of all parameters of this module. """
        params = list(self._params.values())
        for module in self._modules:
            params.extend(self._modules[module].parameters())
        return params

    def forward(self, *inputs: List[Tensor]) -> Tensor:
        raise NotImplementedError

    def __setattr__(self, key, value) -> None:
        """ Register modules and params when assigned. """
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
        input_size,
        output_size,
        initializer: Callable = np.random.random,
        bias=True,
    ):
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
