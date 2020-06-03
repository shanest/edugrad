import itertools
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
        self.fc1 = nn.Linear(input_size, 32)
        self.output = nn.Linear(32, output_size)

    def forward(self, inputs):
        hidden = bcg.relu(self.fc1(inputs))
        return self.output(hidden)


if __name__ == "__main__":

    input_size = 10
    batch_size = 16
    num_epochs = 20

    inputs = np.array(list(itertools.product([0, 1], repeat=input_size))).astype(float)
    # y = sum(x)/2
    targets = np.apply_along_axis(lambda row: np.sum(row) / 2, 1, inputs)[:, np.newaxis]

    model = MLP(input_size, 1)
    optimizer = bcg.optim.SGD(model.parameters())
    data_iterator = bcg.data.BatchIterator()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in data_iterator(inputs, targets):
            model.zero_grad()
            predicted = model(batch.inputs)
            loss = bcg.mse_loss(predicted, batch.targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.value
        print(f'Epoch {epoch} loss: {total_loss / data_iterator.num_batches}')

    loss = bcg.mse_loss(model(bcg.Tensor(inputs, name="x")), bcg.Tensor(targets, name="y"))
    util.draw_graph(bcg.tensor.get_graph_above(loss))