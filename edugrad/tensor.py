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

"""This module defines the Tensor class, which is a wrappers on numpy arrays.

Typically, you will never manually create Tensors.  They are created from 
numpy arrays in two places:
    * in DataIterators (see `data.py`)
    * via `tensor_ops` (see `ops.py`)
"""

from typing import Iterable
import numpy as np
import networkx as nx
import edugrad.ops as ops


class Tensor:
    """Main Tensor class, wrapping a numpy array with gradients etc.

    Attributes:
        value: numpy array with the tensor's value
        parents: Iterable of Tensors, this node's parents in a dynamic graph
        name: a string to name the Tensor
        grad: gradient, if it has been computed; numpy array of same shape as value
    """

    def __init__(self, value: np.ndarray, parents: Iterable = (), name: str = None):
        """Initialize values, set gradients to zero. """
        self.value = value
        self.parents = parents
        self.name = name
        self.grad = np.zeros(value.shape)

    def backward(self) -> None:
        """Run backward pass from a scalar tensor.

        All Tensors in the graph above this one will wind up having their
        gradients stored in `grad`.

        Raises:
            ValueError, if this is not a scalar.
        """
        if not np.isscalar(self.value):
            raise ValueError("Can only call backward() on scalar Tensors.")
        # dL / dL = 1
        self.grad = np.ones(self.value.shape)
        # NOTE: building a graph, then sorting, is not maximally efficient
        # but the graph can be used for visualization etc
        graph = self.get_graph_above()
        reverse_topological = reversed(list(nx.topological_sort(graph)))
        for tensor in reverse_topological:
            tensor._backward()

    def _backward(self) -> None:
        """This is a private helper method.

        Computes the upstream gradients and populates the parent Tensors.
        Pass is the default behavior, which will be evoked only for leaf nodes,
        i.e. those with no parents.

        It gets populated by `ops.tensor_op()`
        """
        pass

    def get_graph_above(self) -> nx.DiGraph:
        """Get the full computation graph of Tensors above the present one. """
        graph = nx.DiGraph()
        visited = set()

        def visit(value: Tensor):
            if value not in visited:
                for parent in value.parents:
                    graph.add_edge(parent, value)
                    visit(parent)
            visited.add(value)

        visit(self)
        return graph

    def __add__(self, other):
        return ops.add(self, other)

    def __radd__(self, other):
        return ops.add(other, self)

    def __sub__(self, other):
        return ops.minus(self, other)