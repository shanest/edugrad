""" See joelnet/data.py by Joel Grus at
https://github.com/joelgrus/joelnet/blob/master/joelnet/data.py """

from typing import NamedTuple, Iterator

import numpy as np

from edugrad.tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator:
    def __call__(self, inputs: np.ndarray, targets: np.ndarray) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: np.ndarray, targets: np.ndarray) -> Iterator[Batch]:
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)
        self.num_batches = len(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = Tensor(inputs[start:end], name="x")
            batch_targets = Tensor(targets[start:end], name="y")
            yield Batch(batch_inputs, batch_targets)
