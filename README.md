# edugrad
 
This is a library intended for pedagogical purposes illustrating a very minimal implementation of dynamic computational graphs with reverse-mode differentiation (backpropagation) for computing gradients.  Three guidelines motivate design choices made in the implementation:
* Mimicking PyTorch's API as closely as possible.
* Simple `forward`/`backward` for operations (operating on numpy arrays).
* Dynamic computation graphs, built as operations are run.

The library has been inspired by several other similar projects.  Specific acknowledgments are in the source where appropriate.
* [`micrograd`](https://github.com/karpathy/micrograd) by Karpathy
* [`autodidact`](https://github.com/mattjj/autodidact): a pedagogical implementation of `autograd`
* [`joelnet`](https://github.com/joelgrus/joelnet)

## Usage

In `examples/toy_half_sum`, you will find a basic use case. `main.py` exhibits a basic use case of defining a feed-forward neural network (multi-layer perceptron) to learn a basic function (in this case, `y = sum(x)/2` where `x` is a binary vector).  You can run it by using `python main.py` from an environment with the packages from `requirements.txt`.

## Basics

There are a few important data structures:
* `Tensor`: this is a wrapper around a numpy array (stored in `.value`), which corresponds to a node in a computation graph, storing information like its parents (if any) and a backward method.
* `Operator`: an operator implements the `forward`/`backward` API and operates directly on numpy arrays.  A decorator `@tensor_op` converts an `Operator` into a method that can be directly called on `Tensor` arguments, which will build the graph dynamically.
* `nn.Module`: as in PyTorch, these are wrappers for graphs that keep track of parameters, sub-modules, etc.