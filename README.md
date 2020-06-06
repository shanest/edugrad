# basic-computational-graph
 
This is a library intended for pedagogical purposes illustrating a very minimal implementation of dynamic computational graphs with reverse-mode differentiation (backpropagation) for computing gradients.  Three guidelines motivate design choices made in the implementation:
* Mimicking PyTorch's API as closely as possible.
* Simple `forward`/`backward` for operations.
* Dynamic computation graphs, built as operations are run.
The library has been inspired by several other similar projects.  Specific acknowledgments are in the source where appropriate.
* [`micrograd`](https://github.com/karpathy/micrograd) by Karpathy
* [`autodidact`](https://github.com/mattjj/autodidact): a pedagogical implementation of `autograd`
* [`joelnet`](https://github.com/joelgrus/joelnet)
