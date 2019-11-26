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
        return np.ones(self._shape) * grad_output * np.prod(self._shape)


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
vec = Variable(np.array([2.0, 1.0]))
mul_op = matmul()
add_op = add()

graph.add_edges_from([(a, mul_op), (vec, mul_op),
                      (mul_op, add_op), (b, add_op)])
nx.draw(graph, graphviz_layout(graph, prog='dot'),
        labels={node: node._name() for node in graph})
plt.show()
run(graph, {"a": np.array([[2.0, 2.0]])})
print(add_op.value)

