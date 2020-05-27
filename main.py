from typing import Dict

import numpy as np
import networkx as nx

import util
import ops
from session import Session

# TODO: write rudimentary trainer
# TODO: simple example (parity?)
# TODO: add README
# TODO: general documentation


if __name__ == "__main__":

    with Session() as sess:
        a = ops.InputNode("a")
        b = ops.InputNode("b")
        diff = ops.minus(a, b)
        util.draw_graph(sess.graph)
        sess.run(diff, {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 1.0])})
        print(diff.value)

    # TODO: write formal tests?
    with Session() as sess:
        a = ops.InputNode("a")
        b = ops.InputNode("b")
        diff = ops.minus(a, b)
        c = ops.InputNode("c")
        add_node = ops.add(diff, c)
        sess.run(
            add_node,
            {
                "a": np.array([1.0, 2.0]),
                "b": np.array([2.0, 1.0]),
                "c": np.array([2.0, 1.0]),
            },
        )
        print(diff.value)
        util.draw_graph(sess.graph)
        sess.backward(add_node)
        print({node._name(): node.grad for node in sess.graph})

    with Session() as sess:
        x = ops.InputNode("x")
        ff1 = ops.feedforward_layer(2, 1, x, ops.relu)
        ff2 = ops.feedforward_layer(1, 1, ff1, ops.relu)

        y = ops.InputNode("y")
        loss_node = ops.mse_loss(ff2, y)

        util.draw_graph(sess.graph)

        sess.run(loss_node, {"x": np.array([[2.0, 2.0]]), "y": np.array([[5.0, 6.0]])})

        print(loss_node.value)
