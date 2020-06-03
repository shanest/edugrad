from typing import List, Callable

import numpy as np
from .tensor import Tensor, Variable


class Operation:
    @staticmethod
    def forward(ctx: List[np.ndarray], *inputs: List[np.ndarray], **kwargs) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: List[np.ndarray], grad_output: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError


def to_function(op: Operation, name: str) -> Callable[[List[Tensor]], Tensor]:
    """
    Takes an operation, and turns it into a callable function on Tensors.

    The resulting function implicitly builds the dynamic computation graph,
    including populating the Tensors' _backward methods, when called.
    """

    def fn(*inputs: List[Tensor], **kwargs) -> Tensor:
        ctx = []
        new_tensor = Tensor(
            op.forward(ctx, *[tensor.value for tensor in inputs], **kwargs), inputs, name
        )

        def _backward():
            grads = op.backward(ctx, new_tensor.grad)
            for idx in range(len(inputs)):
                inputs[idx].grad += grads[idx]

        new_tensor._backward = _backward
        return new_tensor

    return fn


class Add(Operation):
    def forward(ctx, a, b):
        return a + b

    def backward(ctx, grad_output):
        return grad_output, grad_output


add = to_function(Add, "+")


class Minus(Operation):
    def forward(ctx, a, b):
        return a - b

    def backward(ctx, grad_output):
        return grad_output, -grad_output


minus = to_function(Minus, "-")


class MatMul(Operation):
    def forward(ctx, mat1, mat2):
        ctx.append(mat1)
        ctx.append(mat2)
        return mat1 @ mat2

    def backward(ctx, grad_output):
        mat1, mat2 = ctx
        return grad_output @ mat2.T, mat1.T @ grad_output


matmul = to_function(MatMul, "matmul")


class Square(Operation):
    def forward(ctx, a):
        return a ** 2

    def backward(ctx, grad_output):
        return 2 * grad_output


square = to_function(Square, "square")


class ReLU(Operation):
    def forward(ctx, value):
        new_val = np.maximum(0, value)
        ctx.append(new_val)
        return new_val

    def backward(ctx, grad_output):
        value = ctx[-1]
        return (value > 0).astype(float)


relu = to_function(ReLU, "relu")


class ReduceSum(Operation):
    def forward(ctx, value):
        ctx.append(value)
        return np.sum(value)

    def backward(ctx, grad_output):
        return np.ones(ctx[-1].shape) * grad_output


reduce_sum = to_function(ReduceSum, "sum")


class ReduceMean(Operation):
    def forward(ctx, value):
        ctx.append(value)
        return np.mean(value)

    def backward(ctx, grad_output):
        shape = ctx[-1].shape
        return np.ones(shape) * grad_output / np.prod(shape)


reduce_mean = to_function(ReduceMean, "mean")


class CopyRows(Operation):
    """ Copies a (1, dim) array into (num, dim) """
    def forward(ctx, value, num=1):
        return np.tile(value, (num, 1))

    def backward(ctx, grad_output):
        # all rows of grad_output will be the same, so get the first
        return grad_output[0][:, np.newaxis]


copy_rows = to_function(CopyRows, "copy")


def mse_loss(predicted: Tensor, targets: Tensor) -> Tensor:
    diff = minus(predicted, targets)
    squared = square(diff)
    return reduce_mean(squared)


"""
# convert every Operation into a callable function, with its name lowercased
# TODO: decorator for getting doc-string etc?
module_attrs = globals()
for the_op in Operation.__subclasses__():
    name = the_op.__name__.lower()
    module_attrs[name] = to_function(the_op, name)
"""
