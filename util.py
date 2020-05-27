""" Various helper functions. """

from typing import Type, Any, List
import networkx as nx
from ops import InputNode, Variable


def get_nodes_by_type(graph: nx.DiGraph, the_type: Type) -> List[Any]:
    return [node for node in graph if type(node) == the_type]


def get_input_nodes(graph: nx.DiGraph) -> List[Any]:
    return get_nodes_by_type(graph, InputNode)


def get_variables(graph: nx.DiGraph) -> List[Any]:
    return get_nodes_by_type(graph, Variable)
