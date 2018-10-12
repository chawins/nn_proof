import tensorflow as tf
import numpy as np


class Transformer():

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
            # self.transformed = self._transform(
            #     transform_matrix, self.inputs, image_shape)
            # self.transformed = self._transform(
            #     transform_matrix, inputs, image_shape)
            self.transformed = self._transform(
                transform_matrix, self.inputs, image_shape[:-1])

            # MSE between the transformed image and the template
            if hsv:
                # dist = self._hsv_dist(self.transformed, self.templates)
                # self.loss = tf.reduce_sum(dist) / batch_size
                # transformed_hsv = tf.image.rgb_to_hsv(self.transformed)
                # templates_hsv = tf.image.rgb_to_hsv(self.templates)
                # transformed_hsv = tf.image.rgb_to_yuv(self.transformed)
                # templates_hsv = tf.image.rgb_to_yuv(self.templates)
                # dist = self._hsv_dist(transformed_hsv, templates_hsv)
                # self.loss = tf.reduce_sum(dist) / batch_size
                self.loss = tf.reduce_sum(
                    tf.square(self.transformed - self.templates)) / batch_size
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

    def _repeat(self, x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(self, im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = self._repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(self, height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(self, theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = self._meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = self._interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output


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
            if step % 10 == 0:
                print("step: {} - loss: {:.4f}".format(step, loss))

        return self.sess.run(self.transformed, feed_dict=feed_dict)