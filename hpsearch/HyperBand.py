from models.cifar_model import CIFARModel
from abc import ABCMeta, abstractmethod
from math import log2, ceil, floor
import json, os, collections
import tensorflow as tf
import numpy as np


class HyperBand(object):
    __metaclass__ = ABCMeta

    def __init__(self, R, eta, dataset):
        self.R = R
        self.eta = eta
        self.dataset = dataset

    @abstractmethod
    def get_hyperparameter_configuration(self, n):
        pass

    @abstractmethod
    def run_then_return_val_loss(self, model, r_i):
        pass

    @abstractmethod
    def top_k(self, models, keep_ratio):
        pass

    def search(self, debug=True):
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

            models = self.get_hyperparameter_configuration(n)

            for i in xrange(s):
                n_i = floor(n * eta * -i)
                r_i = r * eta ** i

                self.run_then_return_val_loss(models, r_i)
                models = self.top_k(models, keep_ratio=floor(n_i / eta))

        return L


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


class MNISTHyperband(HyperBand):
    def __init__(self, R, eta, dataset, config):
        HyperBand.__init__(self, R, eta, dataset)
        self.default_config = config

    def run_then_return_val_loss(self, models, r_i):
        for model_name in models.keys():
            model = models[model_name][0]

            epoch_to_restart = model.sess.run(tf.train.get_global_step(model.graph))
            model.train(dataset=self.dataset,
                        epoch_to_restart=epoch_to_restart,
                        train_until=r_i)

            models[model_name][1] = model.validation_iter(self.dataset)

    def get_hyperparameter_configuration(self, n):
        models = {}

        def hp_batch_norm():
            return np.random.random() > 0.5

        def hp_lr():
            return np.random.uniform(0.001, 0.000001, 1)

        def hp_dropout():
            return np.random.choice([0.5, 0.7, 0.9, 1])

        def hp_batch_size():
            return np.random.choice([4, 8, 16, 32, 64, 128])

        for _ in range(n):
            new_config = {
                "use_batch_norm": hp_batch_norm(),
                "lr": hp_lr(),
                "keep_prob": hp_dropout(),
                "batch_size": hp_batch_size()
            }
            new_config_name = os.path.basename(self.default_config["result_dir"]) + json.dumps(new_config)
            new_config["result_dir"] = new_config_name

            # Create a new model and build it
            new_model = CIFARModel(new_config)
            graph = tf.Graph()
            new_model.build_graph(graph)

            # Set accuracy to 0.0
            models[new_config_name] = (new_model, 0.0)
        return models

    def top_k(self, models, keep_ratio):
        return sorted(models, key=lambda tup: tup[1])[:keep_ratio]
