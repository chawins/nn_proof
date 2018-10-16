import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
from keras.layers import (Activation, Dense, Flatten, Lambda, Conv2D, Input,
                          MaxPooling2D, Reshape, Concatenate, Cropping2D, Add,
                          Dropout)


class HingeNet():

    def __init__(self, scope, input_shape, output_shape, learning_rate=1e-3, 
                 load_model=True, save_path="model/hingenet.h5"):

        self.scope = scope
        self.save_path = save_path
        self.output_shape = output_shape
        self.height, self.width, self.channel = input_shape

        # Create placeholders
        self.x = tf.placeholder(tf.float32, [None, ] + input_shape, name="x")
        self.y = tf.placeholder(tf.float32, [None, ] + output_shape, name="y")

        # Build model
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            
            inpt = Input(shape=(28, 28, 1))
            conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpt)
            conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
            mp1 = MaxPooling2D(pool_size=(2, 2))(conv2)
            drop1 = Dropout(0.25)(mp1)
            flat = Flatten()(drop1)
            dense1 = Dense(128, activation='relu')(flat)
            drop2 = Dropout(0.5)
            Dense(output_shape[0], activation='softmax')

            self.model = keras.models.Model(inputs=inpt, outputs=output)
            self.output = self.model(self.x)

        # Calculate loss
        scaled_y = 2.*self.y - 1.
        pred = tf.maximum(0., self.n_feats - self.output)
        self.loss = tf.reduce_mean(tf.multiply(scaled_y, pred))

        # Set up optimizer
        with tf.variable_scope(scope + "_opt"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        self.saver = tf.train.Saver(var_list=var_list)
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