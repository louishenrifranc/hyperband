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
        self.nb_iter = self.config['nb_iter']
        self.max_epoch = self.config['max_epoch']
        self.lr = self.config['lr']

        # Batch size is set but we should try to avoid it while building our model
        # because at test time, the first dimension is usually one.
        self.batch_size = self.config['batch_size']

        # Now the child Model needs some custom parameters, to avoid any
        # inheritance hell with the __init__ function, the model
        # will override this function completely
        self.set_model_props()

        # Again, child Model should provide its own build_grap function
        self.graph = self.build_graph(tf.Graph())
        # Init operation: Launch queue if necessary,
        # else initialized variables
        with self.graph.as_default():
            self.init_op = tf.global_variables_initializer()

        # Any operations that should be in the graph but are common to all models
        # can be added this way, here
        with self.graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep=50,
            )

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.writer = tf.summary.FileWriter(self.result_dir, self.sess.graph)

        # This function is not always common to all models, that's why it's again
        # separated from the __init__ one
        self.init()
        # At the end of this function, you want your model to be ready!

    def set_model_props(self):
        # This function is here to be overriden completely.
        # When you look at your model, you want to know exactly which custom options it needs.
        pass

    def build_graph(self, graph):
        raise Exception('The build_graph function must be overriden by the agent')

    def inference_iter(self, inputs):
        raise Exception('The infer function must be overriden by the agent')

    def validation_iter(self, inputs):
        raise Exception('The infer function must be overriden by the agent')

    def learning_iter(self, dataset: Dataset):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The learn_from_epoch function must be overriden by the agent')

    def train(self, dataset: Dataset, save_every=-1, iter_to_restart=-1, iter_to_stop=float("inf")):
        # This function is usually common to all your models, Here is an example:
        for iter_id in range(max(0, iter_to_restart), min(self.nb_iter, iter_to_stop)):
            self.learning_iter(dataset)

            # If you don't want to save during training, you can just pass a negative number
            if save_every > 0 and iter_id % save_every == 0:
                self.save()
        self.save()

    def erase(self):
        if tf.gfile.Exists(self.result_dir):
            import os
            os.remove(self.result_dir)

    def save(self):
        # This function is usually common to all your models, Here is an example:
        global_step = self.sess.run(tf.train.get_global_step(self.graph))

        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.result_dir, global_step))

        self.saver.save(self.sess, self.result_dir + '/model-ep_' + str(global_step), global_step)

        # I always keep the configuration
        if not os.path.isfile(self.result_dir + '/config.json'):
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            # print([n.name for n in self.sess.graph.as_graph_def().node])
            self.sess.run(self.init_op)
        else:

            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
