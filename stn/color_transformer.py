from keras.layers.core import Layer
# from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np


class ColorTransformer(Layer):
    """
    """

    def __init__(self,
                 model,
                 output_size,
                 **kwargs):
        self.model = model
        self.output_size = output_size
        super(ColorTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.model.build(input_shape)
        self.trainable_weights = self.model.trainable_weights

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]),
                int(output_size[1]),
                int(output_size[2]))

    def call(self, X, mask=None):
        weights = self.model.call(X)
        output = self._transform(weights, X)
        return output

    def _transform(self, weights, X):
        return tf.reduce_sum(tf.multiply(X, weights), axis=3, keepdims=True)
