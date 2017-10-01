import os, copy
import tensorflow as tf
import json
from  pprint import pprint

from data.dataset import Dataset


class BasicModel(object):
    # To build your model, you only to pass a "configuration" which is a dictionary
    def __init__(self, config):

        # I make a `deepcopy` of the configuration before using it
        # to avoid any potential mutation when I iterate asynchronously over configurations
        self.config = copy.deepcopy(config)
        if config['debug']:  # This is a personal check i like to do
            pprint('config', self.config)

        # When working with NN, one usually initialize randomly
        # and you want to be able to reproduce your initialization so make sure
        # you store the random seed and actually use it in your TF graph (tf.set_random_seed() for example)
        self.random_seed = self.config['random_seed']
        tf.set_random_seed(self.random_seed)

        # All models share some basics hyper parameters, this is the section where we
        # copy them into the model
        self.result_dir = self.config['result_dir']
        os.makedirs(self.result_dir, exist_ok=True)
        self.nb_iter = self.config['nb_iter']
        self.max_epoch = self.config['max_epoch']
        self.lr = self.config['lr']

        # Batch size is set but we should try to avoid it while building our model
        # because at test time, the first dimension is usually one.
        self.batch_size = self.config['batch_size']

        # Global step

        # Now the child Model needs some custom parameters, to avoid any
        # inheritance hell with the __init__ function, the model
        # will override this function completely
        self.set_model_props()

        # Again, child Model should provide its own build_grap function
        self.build_graph()
        # Init operation: Launch queue if necessary,
        # else initialized variables
        with tf.name_scope(self.config["model_name"]):
            self.global_step = tf.get_variable("global_step", shape=(), dtype=tf.int32, trainable=False)

            all_vars = tf.global_variables()
            self.init_op = tf.variables_initializer(all_vars)
            self.saver = tf.train.Saver(var_list=all_vars, max_to_keep=50)

    def set_model_props(self):
        # This function is here to be overriden completely.
        # When you look at your model, you want to know exactly which custom options it needs.
        pass

    def build_graph(self, graph):
        raise Exception('The build_graph function must be overriden by the agent')

    def inference_iter(self, inputs, sess):
        raise Exception('The infer function must be overriden by the agent')

    def validation_iter(self, inputs, sess):
        raise Exception('The infer function must be overriden by the agent')

    def learning_iter(self, dataset: Dataset, sess):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The learn_from_epoch function must be overriden by the agent')

    def train(self, sess, dataset: Dataset, epoch_to_restart=-1, epoch_to_stop=float("inf")):
        # This function is usually common to all your models, Here is an example:
        for epoch_id in range(max(0, epoch_to_restart), min(self.max_epoch, epoch_to_stop)):
            for _ in range(self.nb_iter):
                self.learning_iter(dataset, sess)
        self.save(sess)

    def erase(self):
        if tf.gfile.Exists(self.result_dir):
            import shutil
            shutil.rmtree(self.result_dir)

    def save(self, sess):
        # This function is usually common to all your models, Here is an example:
        global_step = sess.run(tf.train.get_global_step(tf.get_default_graph()))

        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.result_dir, global_step))

        model = os.path.join(self.result_dir, "model_{}".format(global_step))
        self.saver.save(sess, model, global_step)

        # I always keep the configuration
        config = os.path.join(self.result_dir, "config.json")
        if not os.path.isfile(config):
            with open(config, 'w') as f:
                json.dump(self.config, f)

    def restore(self, sess):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overridden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            print("LLLAAAAAA")
            sess.run(self.init_op)
            return

        if self.config['debug']:
            print('Loading the model from folder: %s' % self.result_dir)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)
