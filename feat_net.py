import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
from keras.layers import (Activation, Dense, Flatten, Lambda, Conv2D, Input,
                          MaxPooling2D, Reshape, Concatenate, Cropping2D, Add,
                          Dropout)
from keras.preprocessing.image import ImageDataGenerator

from stn.spatial_transformer import SpatialTransformer
from stn.conv_model import locnet_v3


class FeatNet():

    def __init__(self, scope, input_shape, output_shape, crop_pos, squeeze=None,
                 hsv=False, thres=None, learning_rate=1e-3, reg=0, 
                 stn_weight=None, load_model=True, 
                 save_path="model/featnet.h5"):

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
        
        # Build model
        self.feat_scores = []
        self.before_sigmoid = []
        inpt = Input(shape=input_shape)
        rescale1 = Lambda(lambda x: x*2 - 1., output_shape=(32, 32, 3))(inpt)
        stn = SpatialTransformer(localization_net=locnet_v3(), 
                                 output_size=(32, 32),
                                 trainable=False,
                                 weights=self.stn_weight)(rescale1)
        v = Lambda(lambda x: x*.5 + .5, output_shape=(32, 32, 3))(stn)

        if squeeze is not None:
            # Simple feature squeezing
            const = squeeze**2 - 1
            v = Lambda(lambda x: tf.cast(x*const, tf.uint8), output_shape=(32, 32, 3))(v)
            v = Lambda(lambda x: tf.cast(x/const, tf.float32), output_shape=(32, 32, 3))(v)

        if hsv:
            # Convert RGB to HSV
            from stn.rgb2hsv import RGB2HSV
            v = RGB2HSV(output_dim=(32, 32, 3))(v)

            if thres is not None:
                thres_type = thres["thres_type"]
                thres_range = thres["thres_range"]
                thres_steep = thres["thres_steep"]

                if thres_type == "diff":
                    from stn.thres import HSVDiffThres
                    v = HSVDiffThres(thres_range, steep=thres_steep, 
                                    output_dim=(32, 32))(v)
                elif thres_type == "hard":
                    from stn.thres import HSVHardThres
                    v = HSVHardThres(thres_range, steep=thres_steep, 
                                    output_dim=(32, 32))(v)

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

            for pos in self.crop_pos:
                top, bot, left, right = pos
                u = Cropping2D(((top, self.height - bot), 
                                (left, self.width - right)))(v)
                u = Conv2D(32, (3, 3), activation="relu")(u)
                u = Conv2D(64, (3, 3), activation="relu")(u)
                u = Flatten()(u)
                u = Dense(128, activation="relu")(u)
                u = Dropout(0.25)(u)
                u = Dense(32, activation="relu")(u)
                u = Dropout(0.5)(u)
                # u = Dense(100, activation="relu")(u)
                u = Dense(1, activation=None)(u)
                self.before_sigmoid.append(u)
                u = Activation("sigmoid")(u)
                # def custom_activation(x):
                #     return x / tf.sqrt(tf.square(x) + 1)
                # u = Activation(custom_activation)(u)
                self.feat_scores.append(u)

            # Define loss
            # 1. Only final feature score
            # with tf.variable_scope("final_layer"):
            #     self.output = Dense(1, activation="sigmoid")(concat)
            # self.loss = tf.losses.mean_squared_error(self.y, self.output)

            # 2. Use naive non-negative constraint on final layer
            # with tf.variable_scope("final_layer"):
            #     self.output = Dense(1, activation="sigmoid", 
            #         kernel_regularizer=keras.regularizers.l2(0.01), 
            #         kernel_constraint=keras.constraints.non_neg())(concat)
            # self.loss = tf.losses.mean_squared_error(self.y, self.output)

            # 3. Penalize negative weights (Lagrangian)
            # with variable_scope("final_layer"):
            #     self.output = Dense(1, activation="sigmoid")(concat)
            # final_layer = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,2
            #                                 scope="featnet/final_layer")
            # tf.minimum()

            # 4. Use softmax on input to last layer

            # 5. Fix final weight to ensure that all features contribute to 
            # the decision
            # self.output = tf.reduce_sum(concat, axis=1, keepdims=True) / self.n_feats)
            # self.loss = tf.losses.mean_squared_error(self.y, self.output)

            # 6. Fix weights + hinge loss, SCORE_THRES = 0.75 (7. SCORE_THRES = 1.)
            before_sigmoid_output = Concatenate()(self.before_sigmoid)
            output = Add()(self.feat_scores)

            self.model = keras.models.Model(inputs=inpt, outputs=output)
            self.model_before_sigmoid = keras.models.Model(
                inputs=inpt, outputs=before_sigmoid_output)
            self.output = self.model(self.x)

        # Weight regularization
        self.reg_loss = 0
        for l in self.model.layers:
            w = l.weights
            if len(w) != 0:
                self.reg_loss += tf.reduce_sum(tf.square(w[0]))

        # Calculate loss
        scaled_y = 2.*self.y - 1.
        pred = tf.maximum(0., self.n_feats - self.output)
        self.loss = tf.reduce_mean(tf.multiply(scaled_y, pred))
        # self.loss = tf.reduce_mean(tf.multiply(scaled_y, -self.output))
        total_loss = self.loss + reg*self.reg_loss

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        # Set up optimizer
        with tf.variable_scope(scope + "_opt"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(total_loss, var_list=var_list)

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
        return self.model_before_sigmoid(x)
        # return self.model(x)

    def train_model(self, sess, data, dataaug=False, n_epoch=10, batch_size=128,
                    thres=0.75):

        x_train, y_train, x_val, y_val = data
        n_train, n_val = len(x_train), len(x_val)

        datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05,
            channel_shift_range=0.1,
            brightness_range=(0.9, 1.1))

        # Initilize all network variables
        sess.run(self.init)

        best_val_loss = 1e9
        n_step = np.ceil(n_train / float(batch_size)).astype(np.int32)
        ind = np.arange(n_train)

        for epoch in range(n_epoch):
            print("============= EPOCH: {} =============".format(epoch))
            # Need to set learning phase to 1 every epoch because model_eval()
            # is also called at the end of every epoch
            K.set_learning_phase(1)
            np.random.shuffle(ind)

            # Training steps
            if not dataaug:
                for step in range(n_step):
                    start = step * batch_size
                    end = (step + 1) * batch_size
                    feed_dict = {self.x: x_train[ind[start:end]], 
                                self.y: y_train[ind[start:end]]}
                    _, loss = sess.run([self.train_op, self.loss], 
                                    feed_dict=feed_dict)
                    if step % 50 == 0:
                        print("STEP: {} \tLoss: {:.4f}".format(step, loss))
            else:
                step = 0
                for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
                    # Using brightness_range in datagen rescales to 255
                    x_batch /= 255.
                    _, loss = sess.run([self.train_op, self.loss], 
                                        feed_dict={self.x: x_batch, self.y: y_batch})
                    if step % 50 == 0:
                        print("STEP: {} \tLoss: {:.4f}".format(step, loss))
                    if step > n_step:
                        break
                    step += 1

            # Print progress
            train_acc, train_loss = self.eval_model(sess, (x_train, y_train), thres=thres)
            val_acc, val_loss = self.eval_model(sess, (x_val, y_val), thres=thres)
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