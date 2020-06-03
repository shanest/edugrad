""" Various helper functions. """

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


def draw_graph(graph: nx.DiGraph) -> None:
    nx.draw(
        graph,
        graphviz_layout(graph, prog="dot"),
        labels={node: node._name() for node in graph},
    )
    # TODO: write to file if specified
    plt.show()
