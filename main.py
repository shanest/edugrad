import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


class Operation:

    def __init__(self, value=None, grad=None, name=None):
        self.value = value
        self.grad = grad
        self.name = name

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, *args):
        value = self.forward(*args)
        self.value = value
        return value

    def _name(self):
        return self.name or type(self).__name__


class LeafOperation(Operation):

    def forward(self):
        return self.value

    def backward(self):
        # TODO: better method here?
        pass


class Variable(LeafOperation):

    def __init__(self, value):
        # no input nodes, value is required
        super(Variable, self).__init__(value=value)


class InputNode(LeafOperation):

    def __init__(self, name):
        super(InputNode, self).__init__(name=name)


class add(Operation):

    def forward(self, a, b):
        return a + b

    def backward(self, output_grad):
        return output_grad, output_grad


class minus(Operation):

    def forward(self, a, b):
        return a - b

    def backward(self, output_grad):
        # TODO: shape?
        return 1.0, -1.0


class matmul(Operation):

    def forward(self, mat1, mat2):
        """ For example:
        mat1: (batch_size, input_size)
        mat2: (input_size, output_size)
        """
        return np.dot(mat1, mat2)

    def backward(self, grad_output):
        pass


class square(Operation):

    def forward(self, value):
        return value**2

    def backward(self, grad_output):
        return 2*grad_output


class scalar_mul(Operation):

    def forward(self, scalar, tensor):
        self._scalar = scalar
        return scalar * tensor

    def backward(self, grad_output):
        return self._scalar * grad_output


class reduce_sum(Operation):

    def forward(self, value: np.ndarray):
        self._shape = value.shape
        return np.sum(value)

    def backward(self, grad_output: np.ndarray):
        return np.ones(self._shape) * grad_output


class reduce_mean(Operation):

    def forward(self, value: np.ndarray):
        self._shape = value.shape
        return np.mean(value)

    def backward(self, grad_output: np.ndarray):
        return np.ones(self._shape) * grad_output * 1 / np.prod(self._shape)


def feedforward_layer(input_size, output_size, input_node: Operation,
                      activation: Operation = None, initializer=np.random.random):

    weights = Variable(initializer((input_size, output_size)))
    biases = Variable(initializer((1, output_size)))
    mul_node = matmul()
    add_node = add()

    edges = [(input_node, mul_node), (weights, mul_node),
             (mul_node, add_node), (biases, add_node)]

    if activation:
        edges.append((add_node, activation()))

    return edges[-1][-1], edges


def get_nodes_by_type(graph, the_type):
    return [node for node in graph if type(node) == the_type]


def get_input_nodes(graph):
    return get_nodes_by_type(graph, InputNode)

def get_variables(graph):
    return get_nodes_by_type(graph, Variable)


def initialize(graph, inputs):
    """ Given a graph with input placeholders and input values, initialize the
    placeholders with the values.

    Note: doesn't do anything fancy.
    """
    input_nodes = get_input_nodes(graph)
    for node in input_nodes:
        node.value = inputs[node.name]


def run(graph, inputs=None):
    initialize(graph, inputs)
    sorted_graph = nx.topological_sort(graph)
    for op in sorted_graph:
        op(*[node.value for node in graph.predecessors(op)])


graph = nx.DiGraph()

# a = Variable(np.array([[2.0, 2.0]]))
a = InputNode("a")
b = Variable(np.array([3.0, 2.0]))
W = Variable(np.array([[2.0], [1.0]]))
mul_op = matmul()
add_op = add()

ff_out_node, ff_edges = feedforward_layer(2, 1, a)
ff2_out_node, ff2_edges = feedforward_layer(1, 1, ff_out_node)

y = InputNode("y")
diff = minus()
square_diff = square()
loss_node = reduce_mean()


graph.add_edges_from(ff_edges + ff2_edges + [
    # TODO: non-linearity here
    (ff2_out_node, diff), (y, diff),
    (diff, square_diff),
    (square_diff, loss_node)
])
nx.draw(graph, graphviz_layout(graph, prog='dot'),
        labels={node: node._name() for node in graph})
plt.show()
run(graph, {"a": np.array([[2.0, 2.0]]),
            "y": np.array([[5.0, 6.0]])})
print(mul_op.value)
print(add_op.value)
print(loss_node.value)
