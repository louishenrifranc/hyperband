from tensorflow.contrib.layers import conv2d, max_pool2d, fully_connected, flatten
import tensorflow.nn.relu as relu
import tensorflow as tf
import numpy as np

from models.basic_model import BasicModel


class CIFARModel(BasicModel):
    def __init__(self, config):
        BasicModel.__init__(self, config)

    def set_model_props(self):
        self.W = self.config['W']
        self.H = self.config['H']

        self.nb_labels = 10

    def build_graph(self, graph):
        with graph.as_default():
            self.input = tf.placeholder(dtype=tf.float32, shape=[-1, self.W, self.H, 3])
            self.labels = tf.placeholder(dtype=tf.int32, shape=[-1])

            with tf.variable_scope("model") as _:
                out = max_pool2d(relu(conv2d(self.input, 64, 3)))
                out = max_pool2d(relu(conv2d(out, 128, 3)))
                out = max_pool2d(relu(conv2d(out, 256, 3)))
                out = relu(fully_connected(flatten(out), 512))

            with tf.variable_scope("output") as _:
                logits = fully_connected(out, self.nb_labels)

            with tf.variable_scope("losses") as _:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                                        logits=logits)

            with tf.variable_scope("accuracy") as _:
                accuracy = tf.reduce_mean(tf.equal(x=self.labels,
                                                   y=tf.arg_max(input=logits,
                                                                dimension=1)))
            with tf.variable_scope("prediction") as _:
                prediction = tf.argmax(logits, axis=1)

            self._cross_entropy = cross_entropy
            self.train_fn = self.optimize()

            self._accuracy = accuracy

            self._prediction = prediction
        return self.graph

    def optimize(self):
        self.config = {}
        optimizer = self.config.get("optimizer", tf.train.AdamOptimizer(learning_rate=self.lr))

        return optimizer.minimize(loss=self._cross_entropy,
                                  var_list=tf.trainable_variables(),
                                  global_step=tf.train.get_global_step())

    def infer(self, inputs):
        with self.sess.as_default() as sess:
            assert np.shape(inputs) == 4
            return sess.run(fetches=self._prediction,
                            feed_dict={
                                self.input: inputs
                            })

    def learn_from_epoch(self, dataset):
        for _ in self.nb_iter:
            with self.sess.as_default() as sess:
                inputs, labels = dataset.get_next_batch()

                sess.run(fetches=self._cross_entropy, feed_dict={
                    self.input: inputs,
                    self.labels: labels
                })

