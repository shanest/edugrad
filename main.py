from typing import Any, Dict

import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

import util
import ops
import config

# TODO: write rudimentary trainer
# TODO: simple example (parity?)
# TODO: add README
# TODO: general documentation


# TODO: should initialize and backward be in session?
def initialize(graph: nx.DiGraph, inputs: Dict[str, np.ndarray]) -> None:
    """ Given a graph with input placeholders and input values, initialize the
    placeholders with the values.

    Note: doesn't do anything fancy.
    """
    input_nodes = util.get_input_nodes(graph)
    for node in input_nodes:
        node.value = inputs[node.name]


def backward(graph: nx.DiGraph, node: ops.Operation) -> None:
    # TODO: instead, reverse the graph, get subgraph at node, topological sort
    # the result of that
    subgraph = get_subgraph_above(graph, node)
    sorted_graph = nx.topological_sort(subgraph)
    reversed_order = reversed(list(sorted_graph))
    node.grad = np.ones(node.value.shape)
    for op in reversed_order:
        if op != node:
            # TODO: separate out zero-ing of grads?
            op.grad = np.zeros(op.value.shape)
            for successor in graph.successors(op):
                grad_successor_wrt_inputs = successor.backward(successor.grad)
                # TODO: refactor so that all backwards return lists?
                if len(grad_successor_wrt_inputs) == 1:
                    op.grad += grad_successor_wrt_inputs
                else:
                    # TODO: there must be a better way of doing this
                    op_index = list(graph.predecessors(successor)).index(op)
                    op.grad += grad_successor_wrt_inputs[op_index]


def draw_graph(graph: nx.DiGraph) -> None:
    nx.draw(
        graph,
        graphviz_layout(graph, prog="dot"),
        labels={node: node._name() for node in graph},
    )
    # TODO: write to file if specified
    plt.show()


def get_subgraph_above(graph: nx.DiGraph, node: Any):
    """ Given a digraph and a node, get the subgraph containing the node and
    all of its ancestors. """
    return nx.subgraph(graph, set([node]) | nx.ancestors(graph, node))


class Session:
    def __enter__(self):
        config._graph = nx.DiGraph()
        self.graph = config._graph
        return self

    def __exit__(self, *exc_details):
        del config._graph

    def run(self, node: ops.Operation, inputs: Dict[str, np.ndarray] = None) -> None:
        subgraph = get_subgraph_above(self.graph, node)
        if inputs:
            initialize(subgraph, inputs)
        sorted_graph = nx.topological_sort(subgraph)
        for op in sorted_graph:
            op(*[node.value for node in subgraph.predecessors(op)])


if __name__ == "__main__":

    with Session() as sess:
        a = ops.InputNode("a")
        b = ops.InputNode("b")
        diff = ops.minus(a, b)
        draw_graph(sess.graph)
        sess.run(diff, {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 1.0])})
        print(diff.value)

    # TODO: write formal tests?
    with Session() as sess:
        a = ops.InputNode("a")
        b = ops.InputNode("b")
        diff = ops.minus(a, b)
        c = ops.InputNode("c")
        add_node = ops.add(diff, c)
        sess.run(
            add_node,
            {
                "a": np.array([1.0, 2.0]),
                "b": np.array([2.0, 1.0]),
                "c": np.array([2.0, 1.0]),
            },
        )
        print(diff.value)
        draw_graph(sess.graph)
        backward(sess.graph, add_node)
        print({node._name(): node.grad for node in sess.graph})

    with Session() as sess:
        x = ops.InputNode("x")
        ff1 = ops.feedforward_layer(2, 1, x, ops.relu)
        ff2 = ops.feedforward_layer(1, 1, ff1, ops.relu)

        y = ops.InputNode("y")
        loss_node = ops.mse_loss(ff2, y)

        draw_graph(sess.graph)

        sess.run(loss_node, {"x": np.array([[2.0, 2.0]]), "y": np.array([[5.0, 6.0]])})

        print(loss_node.value)
