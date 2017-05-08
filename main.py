import os, json, sys
import tensorflow as tf
import random
import time

# See the __init__ script in the models folder
# `make_models` is a helper function to load any models you have
# from models import make_models
# from hpsearch import hyperband, randomsearch

# I personally always like to make my paths absolute
# to be independent from where the python binary is called
dir = os.path.dirname(os.path.realpath(__file__))

# I won't dig into TF interaction with the shell, feel free to explore the documentation
flags = tf.app.flags

# Hyper-parameters search configuration
flags.DEFINE_integer('nb_process', 4, 'Number of parallel process to perform a HP search')
flags.DEFINE_bool('fullsearch', True, 'Explore Hyper Parameter Search')

# fixed_params is a trick I use to be able to fix some parameters inside the model random function
# For example, one might want to explore different models fixing the learning rate, see the basic_model get_random_config function
flags.DEFINE_string('fixed_params', "{}", 'JSON inputs to fix some params in a HP search, ex: \'{"lr": 0.001}\'')

# Agent configuration
flags.DEFINE_string('model_name', 'CIFAR', 'Unique name of the model')

flags.DEFINE_float('keep_prob', 0.7, 'Dropout ratio')
flags.DEFINE_float('lr', 1e-3, 'The learning rate of SGD')
flags.DEFINE_boolean('use_batch_norm', False, 'Batch Norm uses')
flags.DEFINE_integer('batch_size', 32, 'Batch size')

flags.DEFINE_boolean('debug', False, 'Debug mode')
flags.DEFINE_integer('nb_iter', 50, 'Number of training step')
flags.DEFINE_integer('max_epoch', 1000, 'Number of training epoch')

flags.DEFINE_boolean('infer', False, 'Load an agent for playing')

# This is very important for TensorBoard
# each model will end up in its own unique folder using time module
# Obviously one can also choose to name the output folder
flags.DEFINE_string('result_dir', dir + '/results/' + flags.FLAGS.model_name + '/' + str(int(time.time())),
                    'Name of the directory to store/log the model (if it exists, the model will be loaded from it)')

# Another important point, you must provide an access to the random seed
# to be able to fully reproduce an experiment
flags.DEFINE_integer('random_seed', random.randint(0, sys.maxsize), 'Value of random seed')


def main(_):
    config = flags.FLAGS.__flags.copy()

    if config['fullsearch']:
        from hpsearch.HyperBand import MNISTHyperband
        from data.dataset import MNISTDataset
        dataset = MNISTDataset(validation_size=1000)

        cf = MNISTHyperband(R=30,
                            eta=3,
                            dataset=dataset,
                            config=config)
        cf.search()
    else:
        # TODO
        model = make_model(config)

        if config['infer']:
            pass
            # Some code for inference ...
        else:
            pass
            # Some code for training ...


if __name__ == '__main__':
    tf.app.run()
