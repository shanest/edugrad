import numpy as np

import bcg
import bcg.nn as nn

import util

# TODO: write rudimentary trainer
# TODO: simple example (parity?)
# TODO: add README
# TODO: general documentation


class MLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.output = nn.Linear(16, output_size)

    def forward(self, inputs):
        hidden = bcg.relu(self.fc1(inputs))
        return self.output(hidden)


if __name__ == "__main__":

    a = bcg.Variable(np.array([1.0]))
    b = bcg.Variable(np.array([4.0]))
    print(type(a))
    c = bcg.add(a, b)
    print(c.value)
    c.backward()
    print(a.grad)
    print(b.grad)

    model = MLP(5, 3)
    print(isinstance(model, nn.Module))
    print(model._params)
    print(model.parameters())
    output = model(bcg.Tensor(np.random.random((3, 5))))
    print(output.value)
    print(output._backward)
    util.draw_graph(bcg.tensor.get_graph_above(output))