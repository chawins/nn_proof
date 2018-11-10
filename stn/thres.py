from keras.layers.core import Layer
# from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np


class HSVDiffThres(Layer):
    """
    steep = 50 takes about 0.2 to goes from 0 to 1
    """

    def __init__(self,
                 thres_range,
                 output_dim,
                 steep=50,
                 **kwargs):
        self.thres_range = np.array(thres_range)
        self.hwrap = thres_range[0][0] > thres_range[0][1]
        self.output_dim = output_dim
        self.steep = steep
        super(HSVDiffThres, self).__init__(**kwargs)

    def build(self, input_shape):
        super(HSVDiffThres, self).build(input_shape)

    def call(self, x):
        """
        """

        def clip_sigmoid(x):
            return 1 / (1 + tf.exp(-tf.clip_by_value(x, -38, 100)))

        thres_range = tf.constant(self.thres_range, dtype=tf.float32)
        # Soft threshold using two sigmoid
        mask_begin = clip_sigmoid(self.steep*(x - thres_range[:, 0]))
        mask_end = clip_sigmoid(self.steep*(-x + thres_range[:, 1]))
        thres = mask_begin + mask_end - 1
        if self.hwrap:
            const = tf.constant([1, 0, 0], dtype=tf.float32)
            thres = thres + const

        # TODO: Combine to one channel
        # output = tf.reduce_sum(thres, axis=-1) / 3
        output = tf.reduce_prod(thres, axis=-1, keepdims=True)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.output_dim[1], 1)


class HSVHardThres(Layer):
    """
    """

    def __init__(self,
                 thres_range,
                 output_dim,
                 steep=50,
                 **kwargs):
        self.thres_range = np.array(thres_range)
        self.hwrap = thres_range[0][0] > thres_range[0][1]
        self.output_dim = output_dim
        self.steep = steep
        super(HSVHardThres, self).__init__(**kwargs)

    def build(self, input_shape):
        super(HSVHardThres, self).build(input_shape)

    def call(self, x):
        """
        """
        # Hard threshold
        if self.hwrap:
            h = tf.logical_or(x[:, :, :, 0] >= self.thres_range[0, 0], 
                              x[:, :, :, 0] <= self.thres_range[0, 1])
        else:
            h = tf.logical_and(x[:, :, :, 0] >= self.thres_range[0, 0], 
                               x[:, :, :, 0] <= self.thres_range[0, 1])
        s = tf.logical_and(x[:, :, :, 1] >= self.thres_range[1, 0], 
                           x[:, :, :, 1] <= self.thres_range[1, 1])
        v = tf.logical_and(x[:, :, :, 2] >= self.thres_range[2, 0], 
                           x[:, :, :, 2] <= self.thres_range[2, 1])

        h = tf.cast(h, tf.float32)
        s = tf.cast(s, tf.float32)
        v = tf.cast(v, tf.float32)
        return tf.expand_dims(h * s * v, -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.output_dim[1], 1)