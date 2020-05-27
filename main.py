from typing import Tuple, List, Callable, Optional, Any, Type, Dict

import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


class Operation:
    def __init__(self, value=None, grad=None, name=None):
        self.value = value
        self.grad = grad
        self.name = name

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
        # TODO: better method here?
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
    # TODO: non-callable initializers, e.g. np arrays?
    initializer: Callable = np.random.random,
) -> Tuple[Operation, List[Tuple[Operation]]]:

    weights = Variable(initializer((input_size, output_size)), "W")
    biases = Variable(initializer((1, output_size)), "b")
    mul_node = matmul()
    add_node = add()

    edges = [
        (input_node, mul_node),
        (weights, mul_node),
        (mul_node, add_node),
        (biases, add_node),
    ]

    if activation:
        edges.append((add_node, activation()))

    return edges[-1][-1], edges


def mse_loss(
    prediction_node: Operation, target_node: Operation
) -> Tuple[Operation, List[Tuple[Operation]]]:
    diff = minus()
    square_diff = square()
    loss_node = reduce_mean()
    edges = [
        (prediction_node, diff),
        (target_node, diff),
        (diff, square_diff),
        (square_diff, loss_node),
    ]
    return loss_node, edges


def get_nodes_by_type(graph: nx.DiGraph, the_type: Type) -> List[Any]:
    return [node for node in graph if type(node) == the_type]


def get_input_nodes(graph: nx.DiGraph) -> List[Any]:
    return get_nodes_by_type(graph, InputNode)


def get_variables(graph: nx.DiGraph) -> List[Any]:
    return get_nodes_by_type(graph, Variable)


def initialize(graph: nx.DiGraph, inputs: Dict[str, np.ndarray]):
    """ Given a graph with input placeholders and input values, initialize the
    placeholders with the values.

    Note: doesn't do anything fancy.
    """
    input_nodes = get_input_nodes(graph)
    for node in input_nodes:
        node.value = inputs[node.name]


def run(graph: nx.DiGraph, inputs: Dict[str, np.ndarray] = None) -> None:
    if inputs:
        initialize(graph, inputs)
    sorted_graph = nx.topological_sort(graph)
    for op in sorted_graph:
        op(*[node.value for node in graph.predecessors(op)])


def backward(graph: nx.DiGraph, node: Operation) -> None:
    # TODO: raise error if no value?
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


if __name__ == "__main__":

    # TODO: write formal tests?
    test_graph = nx.DiGraph()
    a = InputNode("a")
    b = InputNode("b")
    diff = minus()
    add_node = add()
    c = InputNode("c")
    test_graph.add_edges_from(((a, diff), (b, diff), (diff, add_node), (c, add_node)))
    run(
        test_graph,
        {
            "a": np.array([1.0, 2.0]),
            "b": np.array([2.0, 1.0]),
            "c": np.array([2.0, 1.0]),
        },
    )
    print(diff.value)
    draw_graph(test_graph)
    backward(test_graph, add_node)
    print({node._name(): node.grad for node in test_graph})

    graph = nx.DiGraph()

    x = InputNode("x")
    ff_out_node, ff_edges = feedforward_layer(2, 1, x, relu)
    ff2_out_node, ff2_edges = feedforward_layer(1, 1, ff_out_node, relu)

    y = InputNode("y")
    loss_node, loss_edges = mse_loss(ff2_out_node, y)

    graph.add_edges_from(ff_edges + ff2_edges + loss_edges)
    draw_graph(graph)

    run(graph, {"x": np.array([[2.0, 2.0]]), "y": np.array([[5.0, 6.0]])})

    print(loss_node.value)
