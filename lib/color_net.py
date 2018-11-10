import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
from keras.layers import (Activation, Dense, Flatten, Lambda, Conv2D, Input,
                          MaxPooling2D, Reshape, Concatenate, Cropping2D, Add,
                          Dropout)

from stn.spatial_transformer import SpatialTransformer
from stn.color_transformer import ColorTransformer
from stn.conv_model import locnet_v3


class ColorNet():

    def __init__(self, scope, input_shape, output_shape, crop_pos, 
                 learning_rate=1e-3, stn_weight=None, load_model=True,
                 save_path="model/colornet.h5"):

        self.scope = scope
        self.save_path = save_path
        self.output_shape = output_shape
        self.crop_pos = crop_pos
        self.n_feats = len(crop_pos)
        self.height, self.width, self.channel = input_shape
        self.stn_weight = stn_weight

        # Create placeholders
        self.x = tf.placeholder(tf.float32, [None, ] + input_shape, name="x")
        self.y = tf.placeholder(tf.float32, [None, ] + output_shape, name="y")

        def build_net():
    
            b = np.zeros(4, dtype='float32')
            b[:3] = 0.33
            b[3] = 0
            W = np.zeros((128, 4), dtype='float32')
            weights = [W, b]

            # Regularization
            l2_reg = keras.regularizers.l2(0)

            # Build model
            inpt = Input(shape=(32, 32, 3))
            conv1 = Conv2D(16, (5, 5), padding='same', activation='relu')(inpt)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            conv2 = Conv2D(32, (5, 5), padding='same', activation='relu')(pool1)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            conv3 = Conv2D(64, (5, 5), padding='same', activation='relu')(pool2)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            pool4 = MaxPooling2D(pool_size=(4, 4))(pool1)
            pool5 = MaxPooling2D(pool_size=(2, 2))(pool2)

            flat1 = Flatten()(pool4)
            flat2 = Flatten()(pool5)
            flat3 = Flatten()(pool3)

            merge = Concatenate(axis=-1)([flat1, flat2, flat3])
            dense1 = Dense(128, activation='relu', kernel_regularizer=l2_reg)(merge)
            output = Dense(4, activation=None, kernel_regularizer=l2_reg, weights=weights)(dense1)
            model = keras.models.Model(inputs=inpt, outputs=output)

            return model

        # Build model
        self.feat_scores = []
        self.before_sigmoid = []
        inpt = Input(shape=input_shape)
        rescale1 = Lambda(lambda x: x*2 - 1., output_shape=(32, 32, 3))(inpt)
        color = ColorTransformer(model=build_net(), output_size=(32, 32, 1))(rescale1)

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

    
    
    
            
            self.model = keras.models.Model(inputs=inpt, outputs=output)
            self.output = self.model(self.x)

        # Calculate loss
        self.loss

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        # Set up optimizer
        with tf.variable_scope(scope + "_opt"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss, var_list=var_list)

        opt_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope=scope + "_opt")
        self.init = tf.variables_initializer(var_list=var_list + opt_var_list)

        if load_model:
            try:
                self.model.load_weights(self.save_path)
            except FileNotFoundError:
                print("Saved weights not found...")
                print("Model was built, but no weight was loaded")

    def get_output(self, x):
        return self.model(x)

    def scores(self, sess, x, batch_size=128):

        K.set_learning_phase(0)
        output = np.zeros([len(x), self.n_feats])
        n_step = np.ceil(len(x) / float(batch_size)).astype(np.int32)

        for step in range(n_step):
            start = step * batch_size
            end = (step + 1) * batch_size
            feed_dict = {self.model_before_sigmoid.input: x[start:end]}
            output[start:end] = sess.run(self.model_before_sigmoid.output, 
                                            feed_dict=feed_dict)

        def clip_sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -38, 38)))

        return clip_sigmoid(output)

    def train_model(self, sess, data, n_epoch=10, batch_size=128):

        x_train, y_train, x_val, y_val = data
        n_train, n_val = len(x_train), len(x_val)

        # Initilize all network variables
        sess.run(self.init)

        best_val_loss = 1e9

        for epoch in range(n_epoch):
            print("============= EPOCH: {} =============".format(epoch))
            # Need to set learning phase to 1 every epoch because model_eval()
            # is also called at the end of every epoch
            K.set_learning_phase(1)
            n_step = np.ceil(n_train / float(batch_size)).astype(np.int32)
            ind = np.arange(n_train)
            np.random.shuffle(ind)

            # Training steps
            for step in range(n_step):
                start = step * batch_size
                end = (step + 1) * batch_size
                feed_dict = {self.x: x_train[ind[start:end]], 
                             self.y: y_train[ind[start:end]]}
                _, loss = sess.run([self.train_op, self.loss], 
                                   feed_dict=feed_dict)
                if step % 50 == 0:
                    print("STEP: {} \tLoss: {:.4f}".format(step, loss))

            # Print progress
            train_acc, train_loss = self.eval_model(sess, (x_train, y_train))
            val_acc, val_loss = self.eval_model(sess, (x_val, y_val))
            print("Train Acc|Loss:\t{:.4f}|{:.4f}".format(train_acc, train_loss))
            print("Val Acc|Loss:\t{:.4f}|{:.4f}".format(val_acc, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save model
                self.model.save_weights(self.save_path)

        # Restore to the best saved model
        self.model.load_weights(self.save_path)

    def predict_model(self, sess, x, y=None, batch_size=128):

        K.set_learning_phase(0)
        output = np.zeros([len(x), ] + self.output_shape)
        loss = 0
        n_step = np.ceil(len(x) / float(batch_size)).astype(np.int32)

        for step in range(n_step):
            start = step * batch_size
            end = (step + 1) * batch_size
            if y is None:
                feed_dict = {self.x: x[start:end]}
                output[start:end] = sess.run(self.output, feed_dict=feed_dict)
            else:
                feed_dict = {self.x: x[start:end], self.y: y[start:end]}
                output[start:end], l = sess.run([self.output, self.loss], 
                                                feed_dict=feed_dict)
                loss += l * len(x[start:end])

        if y is None:
            return output 
        else:
            return output, loss / len(x)

    def eval_model(self, sess, data, thres=0.75, batch_size=128):

        x, y = data
        output, loss = self.predict_model(sess, x, y=y, batch_size=batch_size)
        y_thres = output >= self.n_feats * thres
        accuracy = np.mean(np.equal(y_thres, y))
        return accuracy, loss