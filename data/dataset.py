class Dataset(object):
    # To build your model, you only to pass a "configuration" which is a dictionary
    def __init__(self, config):
        self.data = None

    def get_next_batch(self, phase="train"):
        raise Exception('The get_next_batch function must be overriden by the agent')

    @property
    def nb_examples(self, phase="train"):
        if phase in self.data:
            return len(self.data[phase])
        else:
            raise Exception('There is no {} dataset'.format(phase))
