"""This module defines the Tensor class, which is a wrapper on numpy arrays,
and the Operation/tensor_op machinery for building the computation graph.

Typically, you will never manually create Tensors.  They are created from
numpy arrays in two places:
    * in DataIterators (see `data.py`)
    * via `tensor_op` (see `ops.py`)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np


class Operation:
    @staticmethod
    def forward(
        ctx: list[np.ndarray], *args: np.ndarray, **kwargs: dict | None
    ) -> np.ndarray:
        """Forward pass of an operation.

        Args:
            ctx: empty list; can be used to store values for the backward pass
            inputs: arguments to this operation

        Returns:
            output of the operation, assumed to be one numpy array
        """
        raise NotImplementedError

    @staticmethod
    def backward(
        ctx: list[np.ndarray], grad_output: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Backward pass: returns dL/dx for each x in the inputs of this op.

        Args:
            ctx: stored values from the forward pass
            grad_output: dL/dv, where v is the output of this node

        Returns:
            a list of arrays, dL/dx, for each x that was input to this op
        """
        raise NotImplementedError


def tensor_op(op: type[Operation]) -> Callable[[list[Tensor]], Tensor]:
    """Takes an operation and turns it into a callable function on Tensors.

    The resulting function implicitly builds the dynamic computation graph,
    including populating the Tensors' _backward methods, when called.
    """

    def fn(*inputs: Tensor, **kwargs) -> Tensor:
        ctx = []
        new_tensor = Tensor(
            op.forward(ctx, *[tensor.value for tensor in inputs], **kwargs),
            inputs,
            op.__name__,
        )

        def _backward():
            grads = op.backward(ctx, new_tensor.grad)
            for idx, inp in enumerate(inputs):
                inp.grad += grads[idx]

        new_tensor._backward = _backward
        return new_tensor

    return fn


class Tensor:
    """Main Tensor class, wrapping a numpy array with gradients etc.

    Attributes:
        value: numpy array with the tensor's value
        parents: Iterable of Tensors, this node's parents in a dynamic graph
        name: a string to name the Tensor
        grad: gradient, if it has been computed; numpy array of same shape as value
    """

    def __init__(
        self, value: np.ndarray, parents: Iterable = (), name: str | None = None
    ):
        """Initialize values, set gradients to zero."""
        self.value = value
        self.parents = parents
        self.name = name
        self.grad = np.zeros(np.shape(value))

    def backward(self) -> None:
        """Run backward pass from a scalar tensor.

        All Tensors in the graph above this one will wind up having their
        gradients stored in `grad`.

        Raises:
            ValueError: if this is not a scalar.
        """
        if np.ndim(self.value) != 0:
            raise ValueError("Can only call backward() on scalar Tensors.")
        self.grad = np.ones(np.shape(self.value))
        for tensor in reversed(self._topo_sort()):
            tensor._backward()

    def _topo_sort(self) -> list[Tensor]:
        """Return all tensors in the graph above this one in topological order."""
        visited: set[int] = set()
        order: list[Tensor] = []

        def dfs(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                dfs(parent)
            order.append(node)

        dfs(self)
        return order

    def _backward(self) -> None:
        """Compute upstream gradients and accumulate into parent Tensors.

        Default is a no-op for leaf nodes. Populated by `tensor_op()`.
        """
        pass

    def get_graph_above(self) -> list[tuple[Tensor, Tensor]]:
        """Return all edges (parent, child) in the computation graph above this tensor."""
        edges: list[tuple[Tensor, Tensor]] = []
        visited: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                edges.append((parent, node))
                visit(parent)

        visit(self)
        return edges

    def __add__(self, other):
        return ops.add(self, other)

    def __radd__(self, other):
        return ops.add(other, self)

    def __sub__(self, other):
        return ops.minus(self, other)

    def __pow__(self, other):
        return ops.power(self, exponent=other)


# Imported at the bottom to avoid a circular dependency:
# ops.py imports tensor_op from this module, so this module must be
# partially initialized before ops.py can load.
import edugrad.ops as ops  # noqa: E402
