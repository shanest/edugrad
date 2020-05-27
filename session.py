""" A session contains a graph and methods for running ops on the graph. """

from typing import Dict

import networkx as nx
import numpy as np

import config
import ops
import util


class Session:
    def __enter__(self):
        self.graph = nx.DiGraph()
        config._graph = self.graph
        return self

    def __exit__(self, *exc_details):
        del config._graph

    def run(self, node: ops.Operation, inputs: Dict[str, np.ndarray] = None) -> None:
        subgraph = util.get_subgraph_above(self.graph, node)
        if inputs:
            Session._initialize(inputs, subgraph)
        sorted_graph = nx.topological_sort(subgraph)
        for op in sorted_graph:
            op(*[node.value for node in subgraph.predecessors(op)])

    def backward(self, node: ops.Operation):
        subgraph = util.get_subgraph_above(self.graph, node)
        Session._backward(node, subgraph)

    def _initialize(inputs: Dict[str, np.ndarray], graph: nx.DiGraph = None) -> None:
        """ Given a graph with input placeholders and input values, initialize the
        placeholders with the values.

        Note: doesn't do anything fancy.
        """
        input_nodes = util.get_input_nodes(graph)
        for node in input_nodes:
            node.value = inputs[node.name]

    def _backward(node: ops.Operation, graph: nx.DiGraph = None) -> None:
        sorted_graph = nx.topological_sort(graph)
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
