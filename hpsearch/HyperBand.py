from models.mnist_model import MNISTModel
from abc import ABCMeta, abstractmethod
from math import log, ceil, floor
import json, os, collections
from collections import OrderedDict
import tensorflow as tf
import random


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
        # R is the max budget for any configuration. If R = 30, then no
        # configuration will exceed 30 units of time.
        # Here 1 unit of time = 100 iteration = 1 epoch
        R = self.R

        eta = self.eta

        s_max = floor(log(R) / log(3))
        B = (s_max + 1) * R

        if debug:
            print("[DEBUG HYPERBAND] s_max = {}, B = {}".format(s_max, B))

        for s in reversed(range(s_max + 1)):
            n = ceil(B / R / (s + 1) * eta ** s)
            r = R * eta ** -s
            print("[DEBUG HYPERBAND] Iteration s = {} n = {}, r = {}".format(s, n, r))

            models = self.get_hyperparameter_configuration(n)

            for i in range(s):
                n_i = floor(n * eta ** -i)
                r_i = floor(r * eta ** i)
                self.run_then_return_val_loss(models, r_i)
                models = self.top_k(models, keep_ratio=floor(n_i / eta))

        return models


def update(d, u):
    for k, v in d.items():
        if k not in u:
            u[k] = v
    return u


class MNISTHyperband(HyperBand):
    def __init__(self, R, eta, dataset, config):
        HyperBand.__init__(self, R, eta, dataset)
        self.default_config = config

    def run_then_return_val_loss(self, models, r_i):
        for model_name, tup in models.items():
            tf.reset_default_graph()
            model = self.build_model(tup["config"])
            last_epoch = tup["last_epoch"]

            if last_epoch >= r_i:
                continue

            with tf.Session() as sess:
                model.restore(sess=sess)
                model.train(sess=sess,
                            dataset=self.dataset,
                            epoch_to_restart=last_epoch,
                            epoch_to_stop=r_i)

                tup["val_score"] = model.validation_iter(dataset=self.dataset,
                                                         sess=sess)
            tup["last_epoch"] = r_i
            tup["model"] = model

    def get_hyperparameter_configuration(self, n):
        models = {}

        def hp_batch_norm():
            return random.random() > 0.5

        def hp_lr():
            return random.choice([0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.000001])

        def hp_dropout():
            return random.choice([0.5, 0.7, 0.9, 1])

        def hp_batch_size():
            return random.choice([4, 8, 16, 32, 64, 128])

        for _ in range(n):
            new_config = OrderedDict([
                ("use_batch_norm", hp_batch_norm()),
                ("lr", hp_lr()),
                ("keep_prob", hp_dropout()),
                ("batch_size", hp_batch_size())
            ])

            model_unique_name = json.dumps(new_config) \
                .replace("\"", "") \
                .replace("{", "") \
                .replace("}", "") \
                .replace(" ", "") \
                .replace(",", "_") \
                .replace(":", "-")


            # Uniquely defined model name
            new_config["model_name"] = model_unique_name
            new_config["result_dir"] = os.path.join(self.default_config["result_dir"], model_unique_name)
            # Update dictionary of config
            new_config = update(self.default_config, new_config)


            # Set accuracy to 0.0
            models[model_unique_name] = {
                "config": new_config,
                "model": None,
                "last_epoch": 0,
                "val_score": 0.0
            }
        return models

    def top_k(self, models, keep_ratio):
        sorted_model = sorted(models.items(), key=lambda tup: tup[1]["val_score"])

        for model in sorted_model[keep_ratio:]:
            models[model[0]]["model"].erase()
            del models[model[0]]
        return models

    @staticmethod
    def build_model(new_config):
        new_model = MNISTModel(new_config)
        return new_model
