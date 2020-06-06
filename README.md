# basic-computational-graph
 
This is a library intended for pedagogical purposes illustrating a very minimal implementation of dynamic computational graphs with reverse-mode differentiation (backpropagation) for computing gradients.  Three guidelines motivate design choices made in the implementation:
* Mimicking PyTorch's API as closely as possible.
* Simple `forward`/`backward` for operations.
* Dynamic computation graphs, built as operations are run.
The library has been inspired by several other similar projects.  Specific acknowledgments are in the source where appropriate.
* [https://github.com/karpathy/micrograd](`micrograd`) by Karpathy
* [https://github.com/mattjj/autodidact](`autodidact`): a pedagogical implementation of `autograd`
* [https://github.com/joelgrus/joelnet](`joelnet`)