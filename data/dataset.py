import numpy as np
from collections import namedtuple


class Dataset(object):
    # To build your model, you only to pass a "configuration" which is a dictionary
    def __init__(self, data):
        self._data = data

    def get_next_batch(self, phase="train"):
        raise Exception('The get_next_batch function must be overriden by the agent')

    @property
    def nb_examples(self, phase="train"):
        if phase in self._data:
            return len(self._data[phase])
        else:
            raise Exception('There is no {} dataset'.format(phase))


class MNISTDataset(Dataset):
    def __init__(self,
                 validation_ratio):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True, validation_size=validation_ratio)

        data = {
            "train": mnist.train,
            "val": mnist.validation,
            "test": mnist.test
        }
        Dataset.__init__(self, data)


def get_next_batch(self, batch_size, phase="train"):
    assert phase in self._data, "Phase {} is not available".format(phase)

    return self._data[phase].next_batch(batch_size)
