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


class CIFARDataset(Dataset):
    def __init__(self,
                 validation_ratio,
                 test_ratio):
        from data import cifar10

        images, labels = cifar10.distorted_inputs()
        Datum = namedtuple('Datum', 'images label')

        t = test_ratio
        v = validation_ratio
        indexes = np.arange(len(images))
        indexes = np.split(indexes, [int((1 - (t + v)) * len(indexes)), int((1 - t) * len(indexes))])

        data = {
            "train": Datum(images[indexes[0]], labels[indexes[0]]),
            "val": Datum(images[indexes[1]], labels[indexes[1]]),
            "test": Datum(images[indexes[2]], labels[indexes[2]])
        }
        Dataset.__init__(self, data)


def get_next_batch(self, batch_size, phase="train"):
    assert phase in self._data, "Phase {} is not available".format(phase)

    indexes = np.random.randint(0, len(self._data[phase]), batch_size)
    return self._data[phase].images[indexes], self._data[phase].labels[indexes]
