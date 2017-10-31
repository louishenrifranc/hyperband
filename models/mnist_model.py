from tensorflow.contrib.layers import conv2d, max_pool2d, fully_connected, flatten, batch_norm
import tensorflow as tf
import numpy as np

from models.basic_model import BasicModel

relu = tf.nn.relu


class MNISTModel(BasicModel):
    def __init__(self, config):
        BasicModel.__init__(self, config)

    def set_model_props(self):
        self.W = 28
        self.H = 28
        self.nb_labels = 10

        self.use_batch_norm = self.config['use_batch_norm']

    def build_graph(self):
        with tf.name_scope(self.config["model_name"]):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.W * self.H])
            self.labels = tf.placeholder(dtype=tf.int32, shape=[None, self.nb_labels])

            bn_or_id = batch_norm if self.use_batch_norm else tf.identity

            with tf.variable_scope("model") as _:
                input_reshape = tf.reshape(self.input, [-1, self.W, self.H, 1])
                out = max_pool2d(bn_or_id(relu(conv2d(input_reshape, 64, 3))), 2)
                out = max_pool2d(bn_or_id(relu(conv2d(out, 128, 3))), 2)
                out = max_pool2d(bn_or_id(relu(conv2d(out, 256, 3))), 2)
                out = bn_or_id(relu(fully_connected(flatten(out), 512)))

            with tf.variable_scope("output") as _:
                logits = fully_connected(out, self.nb_labels)

            with tf.variable_scope("losses") as _:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                                        logits=logits)

            with tf.variable_scope("accuracy") as _:
                accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(x=tf.arg_max(input=self.labels,
                                              dimension=1),
                                 y=tf.arg_max(input=logits,
                                              dimension=1)),
                        tf.int32))

            with tf.variable_scope("prediction") as _:
                prediction = tf.argmax(logits, axis=1)

            self._cross_entropy = cross_entropy
            self.train_fn = self.optimize()
            self._accuracy = accuracy
            self._prediction = prediction

    def optimize(self):
        optimizer = self.config.get("optimizer", tf.train.AdamOptimizer)(learning_rate=self.lr)

        return optimizer.minimize(loss=self._cross_entropy,
                                  var_list=tf.trainable_variables(),
                                  global_step=tf.train.get_global_step())

    def inference_iter(self, inputs, sess):
        assert np.shape(inputs) == 4
        return sess.run(fetches=self._prediction,
                        feed_dict={
                            self.input: inputs
                        })

    def learning_iter(self, dataset, sess):
        inputs, labels = dataset.get_next_batch(self.batch_size)
        sess.run(fetches=self._cross_entropy, feed_dict={
            self.input: inputs,
            self.labels: labels
        })

    def validation_iter(self, dataset, sess):
        inputs, labels = dataset.get_next_batch(self.batch_size,
                                           phase="val")
        return sess.run(fetches=self._accuracy,
                 feed_dict={
                     self.input: inputs,
                     self.labels: labels
                 })
