from abc import ABCMeta, abstractmethod
from math import log2, ceil, floor


class HyperBand(object):
    __metaclass__ = ABCMeta

    def __init__(self, R, eta=3):
        self.R = R
        self.eta = eta

    @abstractmethod
    def get_hyperparameter_configuration(self, n):
        pass

    @abstractmethod
    def run_then_return_val_loss(self, model, r_i):
        pass

    @abstractmethod
    def top_k(self, models, L, keep_ratio):
        pass

    def search(self, model, debug=True):
        R = self.R
        eta = self.eta

        s_max = floor(log2(R))
        B = (s_max + 1) * R

        if debug:
            from tqdm import trange
            xrange = trange
        else:
            xrange = range
        for s in reversed(xrange(s_max + 1)):
            n = ceil((B * eta ** s) / (R * (s + 1)))
            r = R * eta ** -s

            T = self.get_hyperparameter_configuration(n)

            for i in xrange(s):
                n_i = floor(n * eta * -i)
                r_i = r * eta ** i

                L = self.run_then_return_val_loss(model, r_i)
                T = self.top_k(model, L, keep_ratio=floor(n_i / eta))

        return L


class CIFARHyperband(HyperBand):
    def __init__(self, R, eta):
        HyperBand.__init__(self, R, eta)

    def run_then_return_val_loss(self, model, r_i):
        pass

    def get_hyperparameter_configuration(self, n):
        pass

    def top_k(self, models, L, keep_ratio):
        pass
