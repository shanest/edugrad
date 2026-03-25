"""Defines built-in operations and a helper loss function.

Each operation is a subclass of Operation decorated with @tensor_op, which
converts it into a callable on Tensors that dynamically builds the
computation graph.
"""

import numpy as np

from edugrad.tensor import Operation, tensor_op, Tensor


@tensor_op
class add(Operation):
    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


@tensor_op
class minus(Operation):
    @staticmethod
    def forward(ctx, a, b):
        return a - b

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, -grad_output


@tensor_op
class matmul(Operation):
    @staticmethod
    def forward(ctx, mat1, mat2):
        ctx.append(mat1)
        ctx.append(mat2)
        return mat1 @ mat2

    @staticmethod
    def backward(ctx, grad_output):
        mat1, mat2 = ctx
        return grad_output @ mat2.T, mat1.T @ grad_output


@tensor_op
class power(Operation):
    """Raise to a power, e.g. a^exponent."""

    @staticmethod
    def forward(ctx, a, exponent=1):
        ctx.append(a)
        ctx.append(exponent)
        return a**exponent

    @staticmethod
    def backward(ctx, grad_output):
        value, exponent = ctx
        return (exponent * value ** (exponent - 1) * grad_output,)


@tensor_op
class relu(Operation):
    @staticmethod
    def forward(ctx, value):
        new_val = np.maximum(0, value)
        ctx.append(new_val)
        return new_val

    @staticmethod
    def backward(ctx, grad_output):
        value = ctx[-1]
        return ((value > 0).astype(float) * grad_output,)


@tensor_op
class reduce_sum(Operation):
    @staticmethod
    def forward(ctx, value):
        ctx.append(value)
        return np.sum(value)

    @staticmethod
    def backward(ctx, grad_output):
        return (np.ones(ctx[-1].shape) * grad_output,)


@tensor_op
class reduce_mean(Operation):
    @staticmethod
    def forward(ctx, value):
        ctx.append(value)
        return np.mean(value)

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx[-1].shape
        return (np.ones(shape) * grad_output / np.prod(shape),)


@tensor_op
class copy_rows(Operation):
    """Copies a (dim,) array into (num, dim)."""

    @staticmethod
    def forward(ctx, value, num=1):
        return np.stack([value] * num)

    @staticmethod
    def backward(ctx, grad_output):
        return (np.sum(grad_output, axis=0),)


def mse_loss(predicted: Tensor, targets: Tensor) -> Tensor:
    """Computes mean( (yhat - y)^2 )."""
    diff = minus(predicted, targets)
    squared = diff**2
    return reduce_mean(squared)
