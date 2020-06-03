import numpy as np

import bcg
import bcg.nn as nn

# TODO: write rudimentary trainer
# TODO: simple example (parity?)
# TODO: add README
# TODO: general documentation


if __name__ == "__main__":

    a = bcg.Variable(np.array([1.0]))
    b = bcg.Variable(np.array([4.0]))
    print(type(a))
    c = bcg.add(a, b)
    print(c.value)
    c.backward()
    print(a.grad)
    print(b.grad)

    linear = nn.Linear(5, 3)
    print(linear._params)
    output = linear(bcg.Tensor(np.random.random((3, 5))))
    print(output.value)
    print(output._backward)