""" Defines various operations for neural networks. """

from typing import Optional, Callable
import numpy as np

import config


class Operation:
    def __init__(self, *inputs, value=None, grad=None, name=None):
        # set values
        self.value = value
        self.grad = grad
        self.name = name
        # add node and edges to graph
        config._graph.add_node(self)
        for input_node in inputs:
            config._graph.add_edge(input_node, self)

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, output_grad):
        raise NotImplementedError

    def __call__(self, *args):
        value = self.forward(*args)
        self.value = value
        return value

    def _name(self):
        # TODO: fancy naming for e.g. variables?
        return self.name or type(self).__name__


class LeafOperation(Operation):
    def forward(self):
        return self.value

    def backward(self):
        pass


class Variable(LeafOperation):
    def __init__(self, value, name=None):
        # no input nodes, value is required
        super(Variable, self).__init__(value=value, name=name)


class InputNode(LeafOperation):
    def __init__(self, name):
        super(InputNode, self).__init__(name=name)


class add(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def backward(self, output_grad):
        return output_grad, output_grad


class minus(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    def backward(self, output_grad):
        return output_grad, -output_grad


class matmul(Operation):
    def forward(self, mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
        """ For example:
        mat1: (batch_size, input_size)
        mat2: (input_size, output_size)
        """
        self.mat1 = mat1
        self.mat2 = mat2
        return mat1 @ mat2

    def backward(self, grad_output):
        return grad_output @ self.mat2.T, self.mat1.T @ grad_output


class square(Operation):
    def forward(self, value: np.ndarray) -> np.ndarray:
        return value ** 2

    def backward(self, grad_output):
        return 2 * grad_output


class relu(Operation):
    def forward(self, value: np.ndarray) -> np.ndarray:
        return np.maximum(0, value)

    def backward(self, grad_output):
        # self.value is cached output of last run of forward()
        return (self.value > 0).astype(float)


class reduce_sum(Operation):
    def forward(self, value: np.ndarray) -> np.ndarray:
        self._shape = value.shape
        return np.sum(value)

    def backward(self, grad_output: np.ndarray):
        return np.ones(self._shape) * grad_output


class reduce_mean(Operation):
    def forward(self, value: np.ndarray) -> np.ndarray:
        self._shape = value.shape
        return np.mean(value)

    def backward(self, grad_output: np.ndarray):
        return np.ones(self._shape) * grad_output / np.prod(self._shape)


def feedforward_layer(
    input_size: int,
    output_size: int,
    input_node: Operation,
    activation: Optional[Operation] = None,
    # NOTE: can initialize with a fixed array by using lambda:
    initializer: Callable = np.random.random,
) -> Operation:

    weights = Variable(initializer((input_size, output_size)), "W")
    biases = Variable(initializer((1, output_size)), "b")
    mul_node = matmul(input_node, weights)
    add_node = add(mul_node, biases)

    if activation:
        return activation(add_node)

    return add_node


def mse_loss(prediction_node: Operation, target_node: Operation) -> Operation:
    diff = minus(prediction_node, target_node)
    square_diff = square(diff)
    loss_node = reduce_mean(square_diff)
    return loss_node
