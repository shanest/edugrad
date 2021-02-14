import itertools
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

import edugrad
import edugrad.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, output_size)

    def forward(self, inputs):
        hidden = edugrad.ops.relu(self.fc1(inputs))
        hidden = edugrad.ops.relu(self.fc2(hidden))
        return self.output(hidden)


def draw_graph(graph: nx.DiGraph) -> None:
    nx.draw(
        graph,
        graphviz_layout(graph, prog="dot"),
        labels={node: node.name for node in graph},
    )
    # TODO: write to file if specified
    plt.show()


if __name__ == "__main__":

    input_size = 10
    batch_size = 32
    num_epochs = 20

    # inputs = all binary sequences of length input_size
    inputs = np.array(list(itertools.product([0, 1], repeat=input_size))).astype(float)
    # shuffle inputs before computing targets and splitting
    np.random.shuffle(inputs)
    # y = sum(x)/2
    targets = np.apply_along_axis(lambda row: np.sum(row) / 2, 1, inputs)[:, np.newaxis]

    train_split = int(0.75 * len(inputs))
    train_inputs, train_targets = inputs[:train_split], targets[:train_split]
    test_inputs, test_targets = inputs[train_split:], targets[train_split:]

    model = MLP(input_size, 1)
    optimizer = edugrad.optim.SGD(model.parameters(), lr=1e-3)
    train_iterator = edugrad.data.BatchIterator(batch_size=batch_size)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_iterator(inputs, targets):
            model.zero_grad()
            predicted = model(batch.inputs)
            loss = edugrad.ops.mse_loss(predicted, batch.targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.value
        print(f"Epoch {epoch} loss: {total_loss / train_iterator.num_batches}")

    test_predictions = model(edugrad.tensor.Tensor(test_inputs, name="x"))
    loss = edugrad.ops.mse_loss(
        test_predictions, edugrad.tensor.Tensor(test_targets, name="y")
    )
    print(f"Test loss: {loss.value}")
    draw_graph(loss.get_graph_above())
