from keras.layers.core import Layer
# from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np


class RGB2HSV(Layer):
    """
    """

    def __init__(self,
                 output_dim,
                 **kwargs):
        self.output_dim = output_dim
        super(RGB2HSV, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RGB2HSV, self).build(input_shape)

    def call(self, x):
        """
        Adapted from python code (http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/)
        """
        r, g, b = x[:, :, :, 0], x[:, :, :, 1], x[:, :, :, 2]
        mx = tf.maximum(r, tf.maximum(g, b))
        mn = tf.minimum(r, tf.minimum(g, b))
        df = mx - mn

        # The following block of code implement this:
        # if mx == mn:
        #     h = 0
        # elif mx == r:
        #     h = (60 * ((g-b)/df) + 360) % 360
        # elif mx == g:
        #     h = (60 * ((b-r)/df) + 120) % 360
        # elif mx == b:
        #     h = (60 * ((r-g)/df) + 240) % 360
        
        ZERO = tf.zeros_like(r)
        # This line is necessary for preventing nan gradients from tf.where
        df = tf.where(df > ZERO, df, ZERO + 1)
        # -df < g-b < df and so -1/6 < (g-b)/df < 1/6
        h = tf.where(r > g, 
                     tf.where(r > b, 
                              (g - b)/(df*6) + 1,      # r > b, r > g
                              (r - g)/(df*6) + 2/3),   # b >= r > g
                     tf.where(g > b, 
                              (b - r)/(df*6) + 1/3,    # g > b, g >= r 
                              tf.where(df > ZERO,     # b >= g >= r
                                       (r - g)/(df*6) + 2/3, # b > g >= r
                                       ZERO)))               # b = g = r (df = 0)

        # Ensure circular value falls in [0, 1] correctly (h = h % 360)
        h = tf.where(h > 1, h - 1, h)

        # if mx == 0:
        #     s = 0
        # else:
        #     s = df/mx
        s = tf.where(mx == ZERO, mx, df/mx)

        v = mx
        hsv = tf.concat([tf.expand_dims(h, -1), tf.expand_dims(s, -1), 
                         tf.expand_dims(v, -1)], axis=-1)
        return hsv

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.output_dim