from typing import List, Callable

import numpy as np
from edugrad.tensor import Tensor


class Operation:
    @staticmethod
    def forward(
        ctx: List[np.ndarray], *inputs: List[np.ndarray], **kwargs
    ) -> np.ndarray:
        """Forward pass of an operation.

        Args:
            ctx: empty list of arrays; can be used to store values for backward pass
            inputs: arguments to this operation

        Returns:
            output of the operation, assumed to be one numpy array
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx: List[np.ndarray], grad_output: np.ndarray) -> List[np.ndarray]:
        """Backward pass of an op, returns dL / dx for each x in parents of this op.

        Args:
            ctx: stored values from the forward pass
            grad_output: dL/dv, where v is output of this node

        Returns:
            a _list_ of arrays, dL/dx, for each x that was input to this op
        """
        raise NotImplementedError


def tensor_op(op: Operation) -> Callable[[List[Tensor]], Tensor]:
    """
    Takes an operation, and turns it into a callable function on Tensors.

    The resulting function implicitly builds the dynamic computation graph,
    including populating the Tensors' _backward methods, when called.
    """

    def fn(*inputs: List[Tensor], **kwargs) -> Tensor:
        ctx = []
        new_tensor = Tensor(
            op.forward(ctx, *[tensor.value for tensor in inputs], **kwargs),
            inputs,
            op.__name__,
        )

        def _backward():
            grads = op.backward(ctx, new_tensor.grad)
            for idx in range(len(inputs)):
                inputs[idx].grad += grads[idx]

        new_tensor._backward = _backward
        return new_tensor

    return fn


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
    """Raise to a power, e.g. a^exponent. """
    @staticmethod
    def forward(ctx, a, exponent = 1):
        ctx.append(a)
        ctx.append(exponent)
        return a ** exponent

    @staticmethod
    def backward(ctx, grad_output):
        value, exponent = ctx
        return [exponent * value ** (exponent-1) * grad_output]


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
        return [(value > 0).astype(float) * grad_output]


@tensor_op
class reduce_sum(Operation):
    @staticmethod
    def forward(ctx, value):
        ctx.append(value)
        return np.sum(value)

    @staticmethod
    def backward(ctx, grad_output):
        return [np.ones(ctx[-1].shape) * grad_output]


@tensor_op
class reduce_mean(Operation):
    @staticmethod
    def forward(ctx, value):
        ctx.append(value)
        return np.mean(value)

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx[-1].shape
        return [np.ones(shape) * grad_output / np.prod(shape)]


@tensor_op
class copy_rows(Operation):
    """ Copies a (dim) array into (num, dim) """

    @staticmethod
    def forward(ctx, value, num=1):
        return np.stack([value] * num)

    @staticmethod
    def backward(ctx, grad_output):
        return [np.sum(grad_output, axis=0)]


def mse_loss(predicted: Tensor, targets: Tensor) -> Tensor:
    """Computes mean( (yhat - y)^2 ) """
    diff = minus(predicted, targets)
    squared = diff**2
    return reduce_mean(squared)