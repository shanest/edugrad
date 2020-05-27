from typing import List, Callable, Optional, Any, Type, Dict

import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

# TODO: split into files!
# TODO: write rudimentary trainer
# TODO: simple example (parity?)
# TODO: add README
# TODO: general documentation


class Operation:
    def __init__(self, *inputs, value=None, grad=None, name=None):
        # set values
        self.value = value
        self.grad = grad
        self.name = name
        # add node and edges to graph
        _graph.add_node(self)
        for input_node in inputs:
            _graph.add_edge(input_node, self)

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, output_grad):
        raise NotImplementedError

    def __call__(self, *args):
        value = self.forward(*args)
        self.value = value
        return value

    def _name(self):
        # TODO: fancy naming for e.g. variables?
        return self.name or type(self).__name__


class LeafOperation(Operation):
    def forward(self):
        return self.value

    def backward(self):
        pass


class Variable(LeafOperation):
    def __init__(self, value, name=None):
        # no input nodes, value is required
        super(Variable, self).__init__(value=value, name=name)


class InputNode(LeafOperation):
    def __init__(self, name):
        super(InputNode, self).__init__(name=name)


class add(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def backward(self, output_grad):
        return output_grad, output_grad


class minus(Operation):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    def backward(self, output_grad):
        # TODO: shape?
        return output_grad, -output_grad


class matmul(Operation):
    def forward(self, mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
        """ For example:
        mat1: (batch_size, input_size)
        mat2: (input_size, output_size)
        """
        return np.dot(mat1, mat2)

    def backward(self, grad_output):
        # TODO: implement this!
        pass


class square(Operation):
    def forward(self, value: np.ndarray) -> np.ndarray:
        return value ** 2

    def backward(self, grad_output):
        return 2 * grad_output


class relu(Operation):
    def forward(self, value: np.ndarray) -> np.ndarray:
        return np.maximum(0, value)

    def backward(self, grad_output):
        # self.value is cached output of last run of forward()
        return (self.value > 0).astype(float)


class reduce_sum(Operation):
    def forward(self, value: np.ndarray) -> np.ndarray:
        self._shape = value.shape
        return np.sum(value)

    def backward(self, grad_output: np.ndarray):
        return np.ones(self._shape) * grad_output


class reduce_mean(Operation):
    def forward(self, value: np.ndarray) -> np.ndarray:
        self._shape = value.shape
        return np.mean(value)

    def backward(self, grad_output: np.ndarray):
        return np.ones(self._shape) * grad_output / np.prod(self._shape)


def feedforward_layer(
    input_size: int,
    output_size: int,
    input_node: Operation,
    activation: Optional[Operation] = None,
    # NOTE: can initialize with a fixed array by using lambda:
    initializer: Callable = np.random.random,
) -> Operation:

    weights = Variable(initializer((input_size, output_size)), "W")
    biases = Variable(initializer((1, output_size)), "b")
    mul_node = matmul(input_node, weights)
    add_node = add(mul_node, biases)

    if activation:
        return activation(add_node)

    return add_node


def mse_loss(prediction_node: Operation, target_node: Operation) -> Operation:
    diff = minus(prediction_node, target_node)
    square_diff = square(diff)
    loss_node = reduce_mean(square_diff)
    return loss_node


def get_nodes_by_type(graph: nx.DiGraph, the_type: Type) -> List[Any]:
    return [node for node in graph if type(node) == the_type]


def get_input_nodes(graph: nx.DiGraph) -> List[Any]:
    return get_nodes_by_type(graph, InputNode)


def get_variables(graph: nx.DiGraph) -> List[Any]:
    return get_nodes_by_type(graph, Variable)


# TODO: should initialize and backward be in session?
def initialize(graph: nx.DiGraph, inputs: Dict[str, np.ndarray]) -> None:
    """ Given a graph with input placeholders and input values, initialize the
    placeholders with the values.

    Note: doesn't do anything fancy.
    """
    input_nodes = get_input_nodes(graph)
    for node in input_nodes:
        node.value = inputs[node.name]


def backward(graph: nx.DiGraph, node: Operation) -> None:
    # TODO: instead, reverse the graph, get subgraph at node, topological sort
    # the result of that
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


def draw_graph(graph: nx.DiGraph) -> None:
    nx.draw(
        graph,
        graphviz_layout(graph, prog="dot"),
        labels={node: node._name() for node in graph},
    )
    # TODO: write to file if specified
    plt.show()


class Session:
    def __enter__(self):
        global _graph
        _graph = nx.DiGraph()
        self.graph = _graph
        return self

    def __exit__(self, *exc_details):
        global _graph
        del _graph

    def run(self, node: Operation, inputs: Dict[str, np.ndarray] = None) -> None:
        if inputs:
            initialize(self.graph, inputs)
        sorted_graph = nx.topological_sort(self.graph)
        for op in sorted_graph:
            op(*[node.value for node in self.graph.predecessors(op)])


if __name__ == "__main__":

    with Session() as sess:
        a = InputNode("a")
        b = InputNode("b")
        diff = minus(a, b)
        draw_graph(sess.graph)
        sess.run(diff, {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 1.0])})
        print(diff.value)

    # TODO: write formal tests?
    with Session() as sess:
        a = InputNode("a")
        b = InputNode("b")
        diff = minus(a, b)
        c = InputNode("c")
        add_node = add(diff, c)
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
        x = InputNode("x")
        ff1 = feedforward_layer(2, 1, x, relu)
        ff2 = feedforward_layer(1, 1, ff1, relu)

        y = InputNode("y")
        loss_node = mse_loss(ff2, y)

        draw_graph(sess.graph)

        sess.run(loss_node, {"x": np.array([[2.0, 2.0]]), "y": np.array([[5.0, 6.0]])})

        print(loss_node.value)
