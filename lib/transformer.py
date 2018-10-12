from keras.layers.core import Layer
import tensorflow as tf
import numpy as np


class Transformer():
    """
    Code is adapted from: https://github.com/hello2all/GTSRB_Keras_STN/blob/
    master/spatial_transformer.py
    """

    def __init__(self, image_shape, sess=None, batch_size=1, learning_rate=1e-3,
                 hsv=True):

        self.image_shape = image_shape
        shape = np.concatenate([np.array([batch_size]), image_shape])
        if sess:
            self.sess = sess
        else:
            self.sess = tf.Session()

        # Set up placeholder for templates and input images
        self.templates = tf.placeholder(tf.float32, shape=shape,
                                        name="transformer_templates")
        self.inputs = tf.placeholder(tf.float32, shape=shape,
                                     name="transformer_inputs")

        # Set initialized value for transformation matrix
        b = np.zeros([batch_size, 2, 3], dtype=np.float32)
        b[:, 0, 0] = 1
        b[:, 1, 1] = 1
        matrix_init = tf.constant(b, dtype=tf.float32)

        # Create transformation matrix variable
        with tf.variable_scope("Transformer", reuse=tf.AUTO_REUSE):
            # Extra variable to calibrate brightness
            # zeros = np.zeros([batch_size, 1, 1, 1], dtype=np.float32)
            # offset_init = tf.constant(zeros, tf.float32)
            # self.offset = tf.get_variable("offset", dtype=tf.float32, 
            #                               initializer=offset_init, 
            #                               trainable=True)
            # inputs = self.inputs + self.offset

            transform_matrix = tf.get_variable("transform_matrix",
                                               dtype=tf.float32,
                                               initializer=matrix_init,
                                               trainable=True)
            # Transform inputs using the matrix
            self.transformed = self._transform(
                transform_matrix, self.inputs, image_shape)
            # self.transformed = self._transform(
            #     transform_matrix, inputs, image_shape)
        # Set initialized value for transformation matrix
        # b = np.zeros([batch_size, 2, 3], dtype=np.float32)
        # b[:, 0, 0] = 1
        # b[:, 1, 1] = 1
        # b = b.reshape([batch_size, 6])
        # matrix_init = tf.constant(b, dtype=tf.float32)

        # # Create transformation matrix variable
        # with tf.variable_scope("Transformer", reuse=tf.AUTO_REUSE):
        #     transform_matrix = tf.get_variable("transform_matrix",
        #                                        dtype=tf.float32,
        #                                        initializer=matrix_init,
        #                                        trainable=True)
        #     zeros = tf.zeros([batch_size, 6])
        #     transform = tf.concat([transform_matrix, zeros], axis=1)
        #     self.transformed = tf.contrib.image.transform(
        #         self.inputs, transform, interpolation='BILINEAR')

            # MSE between the transformed image and the template
            if hsv:
                dist = self._hsv_dist(self.transformed, self.templates)
                self.loss = tf.reduce_sum(dist) / batch_size
                # transformed_hsv = tf.image.rgb_to_hsv(self.transformed)
                # templates_hsv = tf.image.rgb_to_hsv(self.templates)
                # transformed_hsv = tf.image.rgb_to_yuv(self.transformed)
                # templates_hsv = tf.image.rgb_to_yuv(self.templates)
                # dist = self._hsv_dist(transformed_hsv, templates_hsv)
                # self.loss = tf.reduce_sum(dist) / batch_size
            else:
                self.loss = tf.reduce_sum(
                    tf.square(self.transformed - self.templates)) / batch_size
                # self.loss = tf.reduce_sum(
                #     tf.abs(self.transformed - self.templates)) / batch_size

            # Set up optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # self.update_step = optimizer.minimize(
            #     self.loss, var_list=[transform_matrix, self.offset])
            self.update_step = optimizer.minimize(
                self.loss, var_list=[transform_matrix])
            # var_list = [x for x in tf.global_variables()
            #             if "Transformer" in x.name]
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope="Transformer")
            self.init = tf.variables_initializer(var_list=var_list)

    def _hsv_dist(self, image_a, image_b):
        dist1 = tf.square(image_a - image_b)
        dist2 = tf.square(image_a + 1 - image_b)
        dist3 = tf.square(image_a - image_b - 1)
        dist = tf.minimum(dist1, dist2)
        return tf.minimum(dist, dist3)

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1, 1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        num_channels = tf.shape(image)[3]

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        height_float = tf.cast(height, tf.float32)
        width_float = tf.cast(width, tf.float32)

        output_height = output_size[0]
        output_width = output_size[1]

        x = .5*(x + 1.0)*(width_float)
        y = .5*(y + 1.0)*(height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype=tf.float32)
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a*pixel_values_a,
                           area_b*pixel_values_b,
                           area_c*pixel_values_c,
                           area_d*pixel_values_d])
        return output

    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, shape=(1, -1))
        y_coordinates = tf.reshape(y_coordinates, shape=(1, -1))
        ones = tf.ones_like(x_coordinates)
        indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
        return indices_grid

    def _transform(self, affine_transformation, input_shape, output_size):
        batch_size = tf.shape(input_shape)[0]
        height = tf.shape(input_shape)[1]
        width = tf.shape(input_shape)[2]
        num_channels = tf.shape(input_shape)[3]

        affine_transformation = tf.reshape(
            affine_transformation, shape=(batch_size, 2, 3))
        affine_transformation = tf.cast(affine_transformation, tf.float32)

        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)
        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1])  # flatten?
        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, tf.stack([batch_size, 3, -1]))

        # transformed_grid = tf.batch_matmul(affine_transformation, indices_grid)
        transformed_grid = tf.matmul(affine_transformation, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x_s_flatten = tf.reshape(x_s, [-1])
        y_s_flatten = tf.reshape(y_s, [-1])
        self.transformed_grid = transformed_grid

        transformed_image = self._interpolate(input_shape,
                                              x_s_flatten,
                                              y_s_flatten,
                                              output_size)

        transformed_image = tf.reshape(transformed_image, shape=(batch_size,
                                                                 output_height,
                                                                 output_width,
                                                                 num_channels))
        return transformed_image

    def transform(self, inputs, templates, n_steps=50):
        """
        Uses a gradient-descent optimizer to try to match inputs to templates.
        Returns transformed input images.
        Shape is (None, height, width, channels).

        :param inputs: Numpy array of input images
        :param templates: Numy array of templates to match inputs to
        :param n_steps: (optional) Number of steps to take in the optimization
        """

        self.sess.run(self.init)
        feed_dict = {self.templates: templates, self.inputs: inputs}

        for step in range(n_steps):
            _, loss = self.sess.run([self.update_step, self.loss],
                                    feed_dict=feed_dict)
            # loss = self.sess.run(self.loss, feed_dict=feed_dict)
            # print(self.sess.run(self.transformed_grid, feed_dict=feed_dict))
            if step % 10 == 0:
                print("step: {} - loss: {:.4f}".format(step, loss))

        return self.sess.run(self.transformed, feed_dict=feed_dict)

# TODO: add learning rate decay
