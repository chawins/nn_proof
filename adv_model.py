import functools

import tensorflow as tf
from cleverhans.model import Model


class CnnGtsrb(Model):
    def __init__(self, scope, nb_classes, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())

        # Do a dummy run of fprop to make sure the variables are created from
        # the start
        self.fprop(tf.placeholder(tf.float32, [128, 32, 32, 3]))
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs
        my_conv = functools.partial(tf.layers.conv2d, activation=tf.nn.relu,
                                    kernel_initializer=tf.glorot_normal_initializer())
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            y = tf.keras.layers.Lambda(lambda x: x*2 - 1.)(x)
            y = my_conv(y, 16, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = my_conv(y, 32, 6, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = my_conv(y, 64, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = my_conv(y, 96, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = my_conv(y, 128, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = my_conv(y, 192, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = tf.layers.max_pooling2d(
                y, pool_size=2, strides=2, padding="same")
            y = my_conv(y, 256, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = tf.layers.max_pooling2d(
                y, pool_size=2, strides=2, padding="same")
            y = my_conv(y, 128, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = my_conv(y, 64, 5, strides=1, padding="same")
            y = tf.layers.max_pooling2d(
                y, pool_size=8, strides=8, padding="same")
            logits = tf.layers.dense(
                tf.layers.flatten(y), 43,
                kernel_initializer=tf.glorot_normal_initializer())
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}


class CnnGtsrbV1(Model):
    def __init__(self, scope, nb_classes, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())

        # Do a dummy run of fprop to make sure the variables are created from
        # the start
        self.fprop(tf.placeholder(tf.float32, [128, 32, 32, 3]))
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs
        my_conv = functools.partial(tf.layers.conv2d, activation=tf.nn.relu,
                                    kernel_initializer=tf.glorot_normal_initializer())
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            y = tf.keras.layers.Lambda(lambda x: x*2 - 1.)(x)
            y = my_conv(y, 32, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = my_conv(y, 64, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = my_conv(y, 128, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = tf.layers.max_pooling2d(
                y, pool_size=2, strides=2, padding="same")
            y = my_conv(y, 128, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = tf.layers.max_pooling2d(
                y, pool_size=2, strides=2, padding="same")
            y = my_conv(y, 64, 5, strides=1, padding="same")
            y = tf.layers.max_pooling2d(
                y, pool_size=2, strides=2, padding="same")
            y = tf.layers.dense(
                tf.layers.flatten(y), 1024,
                kernel_initializer=tf.glorot_normal_initializer())
            y = tf.layers.dense(
                y, 43, kernel_initializer=tf.glorot_normal_initializer())
            return {self.O_LOGITS: y,
                    self.O_PROBS: tf.nn.softmax(logits=y)}


class CnnGtsrbV2(Model):
    def __init__(self, scope, nb_classes, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())

        # Do a dummy run of fprop to make sure the variables are created from
        # the start
        self.fprop(tf.placeholder(tf.float32, [128, 32, 32, 3]))
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs

        reg = tf.contrib.layers.l2_regularizer(1e-2)

        my_conv = functools.partial(tf.layers.conv2d, activation=tf.nn.relu,
                                    kernel_initializer=tf.glorot_normal_initializer(),
                                    kernel_regularizer=reg)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            y = tf.keras.layers.Lambda(lambda x: x*2 - 1.)(x)
            y = my_conv(y, 32, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = my_conv(y, 64, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = my_conv(y, 128, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = tf.layers.max_pooling2d(
                y, pool_size=2, strides=2, padding="same")
            y = my_conv(y, 128, 5, strides=1, padding="same")
            y = tf.layers.batch_normalization(y)
            y = tf.layers.max_pooling2d(
                y, pool_size=2, strides=2, padding="same")
            y = my_conv(y, 64, 5, strides=1, padding="same")
            y = tf.layers.max_pooling2d(
                y, pool_size=2, strides=2, padding="same")
            y = tf.layers.dense(
                tf.layers.flatten(y), 1024,
                kernel_initializer=tf.glorot_normal_initializer(),
                kernel_regularizer=reg)
            y = tf.layers.dropout(y, 0.25)
            y = tf.layers.dense(
                y, 43, kernel_initializer=tf.glorot_normal_initializer())
            y = tf.layers.dropout(y, 0.5)
            return {self.O_LOGITS: y,
                    self.O_PROBS: tf.nn.softmax(logits=y)}
