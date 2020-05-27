""" Various helper functions. """

from typing import Type, Any, List

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

from ops import InputNode, Variable


def get_nodes_by_type(graph: nx.DiGraph, the_type: Type) -> List[Any]:
    return [node for node in graph if type(node) == the_type]


def get_input_nodes(graph: nx.DiGraph) -> List[Any]:
    return get_nodes_by_type(graph, InputNode)


def get_variables(graph: nx.DiGraph) -> List[Any]:
    return get_nodes_by_type(graph, Variable)


def get_subgraph_above(graph: nx.DiGraph, node: Any):
    """ Given a digraph and a node, get the subgraph containing the node and
    all of its ancestors. """
    return nx.subgraph(graph, set([node]) | nx.ancestors(graph, node))


def draw_graph(graph: nx.DiGraph) -> None:
    nx.draw(
        graph,
        graphviz_layout(graph, prog="dot"),
        labels={node: node._name() for node in graph},
    )
    # TODO: write to file if specified
    plt.show()
