from typing import Iterable
import numpy as np
import networkx as nx


class Tensor:
    def __init__(self, value: np.ndarray, parents: Iterable = (), name: str = None):
        self.value = value
        self.parents = parents
        self.name = name
        self.grad = np.zeros(value.shape)

    def backward(self) -> None:
        self.grad = np.ones(self.value.shape)
        backprop(self)

    def _backward(self) -> None:
        raise NotImplementedError("This Tensor has no _backward implemented")


class Variable(Tensor):
    def __init__(self, value, name=None):
        super(Variable, self).__init__(value, name=name)

    def _backward(self) -> None:
        pass


def get_graph_above(node: Tensor) -> nx.DiGraph:
    graph = nx.DiGraph()
    visited = set()

    def visit(value: Tensor):
        if value not in visited:
            for parent in value.parents:
                graph.add_edge(parent, value)
                visit(parent)
        visited.add(value)

    visit(node)
    return graph


def backprop(node: Tensor) -> None:
    # NOTE: building a graph, then sorting, is not maximally efficient
    # but the graph can be used for visualization etc
    graph = get_graph_above(node)
    reverse_topological = reversed(list(nx.topological_sort(graph)))
    for tensor in reverse_topological:
        tensor._backward()
