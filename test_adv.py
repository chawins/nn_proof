import json
import logging
import os
import time
import pickle
from keras import backend as K

from cleverhans.attacks import CarliniWagnerL2, FastGradientMethod, MadryEtAl
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval
from lib.utils import load_gtsrb
from parameters import *
from small_net import *
# Load and set up all models
from stn.conv_model import build_cnn_no_stn, conv_model_no_color_adjust

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def build_mnist():

    from keras.layers import (Activation, Dense, Dropout, Flatten, Lambda,
                              MaxPooling2D, Reshape)

    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    return model


def build_gtsrb():

    # from keras.layers import (Activation, Dense, Dropout, Flatten, Lambda,
    #                           MaxPooling2D, Reshape, BatchNormalization)
    from keras.layers import Reshape

    model = Sequential()
    model.add(Lambda(
        lambda x: x*2 - 1.,
        input_shape=(32, 32, 3),
        output_shape=(32, 32, 3)))
    # model.add(Reshape((32, 32, 3), input_shape=(32, 32, 3)))
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(96, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(192, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Flatten())
    # model.add(Dropout(0.6))
    model.add(Dense(43, activation='softmax'))

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    return model


def eval_cleverhans_clean(X_np, y_np, batch_size=128):
    y_probs = np.zeros_like(y_np)
    loss = 0.
    for i in range(len(X_np) // batch_size):
        start = i*batch_size
        end = (i + 1)*batch_size
        feed_dict = {x: X_np[start:end], y: y_np[start:end]}
        y_probs[start:end] = sess.run(logits_clean, feed_dict=feed_dict)
        loss += sess.run(loss_clean, feed_dict=feed_dict)
    y_probs[end:] = sess.run(logits_clean, feed_dict={x: X_np[end:]})
    loss += sess.run(loss_clean, feed_dict={x: X_np[end:], y: y_np[end:]})
    acc = np.sum(np.argmax(y_probs, axis=1) ==
                 np.argmax(y_np, axis=1)) / len(X_np)
    loss /= len(X_np)
    return loss, acc


def eval_cleverhans_adv(X_np, y_np, batch_size=128):
    y_probs = np.zeros_like(y_np)
    loss = 0.
    for i in range(len(X_np) // batch_size):
        start = i*batch_size
        end = (i + 1)*batch_size
        feed_dict = {x: X_np[start:end], y: y_np[start:end]}
        y_probs[start:end] = sess.run(logits_adv, feed_dict=feed_dict)
        loss += sess.run(loss_adv, feed_dict=feed_dict)
    y_probs[end:] = sess.run(logits_adv, feed_dict={x: X_np[end:]})
    loss += sess.run(loss_adv, feed_dict={x: X_np[end:], y: y_np[end:]})
    acc = np.sum(np.argmax(y_probs, axis=1) ==
                 np.argmax(y_np, axis=1)) / len(X_np)
    loss /= len(X_np)
    return loss, acc

# Object used to keep track of (and return) key accuracies
# report = AccuracyReport()


# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

if K.image_dim_ordering() != 'tf':
    K.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

# Create TF session and set as Keras backend session
# config_sess = tf.ConfigProto()
# config_sess.gpu_options.allow_growth = True
# sess = tf.Session(config=config_sess)
sess = tf.Session()
K.set_session(sess)

# set_log_level(logging.DEBUG)

with open('config.json') as config_file:
    config = json.load(config_file)
dataset = config['dataset']
weights_path = config['weights_path']
model_name = config['model_name']
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
batch_size = config['training_batch_size']

# Get TF logger
log = logging.getLogger('test_adv')
log.setLevel(logging.DEBUG)
# Create formatter and add it to the handlers
formatter = logging.Formatter(
    '[%(levelname)s %(asctime)s %(name)s] %(message)s')
# Create file handler
fh = logging.FileHandler(model_name + ".log", mode='w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

if dataset == 'gtsrb':
    # Load GTSRB dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_gtsrb()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    # Obtain Image Parameters
    n_train, img_rows, img_cols, nchannels = X_train.shape
    nb_classes = y_train.shape[1]

    # model = build_gtsrb()
    # model = build_cnn_no_stn()
    # model = conv_model_no_color_adjust()
    from adv_model import *
    model = CnnGtsrbV1('CnnGtsrbV1', nb_classes)
    # model.summary()

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

elif dataset == 'mnist':
    # Load MNIST dataset
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.
    X_test = X_test / 255.
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)
    n_train, img_rows, img_cols = X_train.shape
    nb_classes = y_train.shape[1]

    model = build_mnist()

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

# Set up adversarial example generation using cleverhans
# wrap_model = KerasModelWrapper(model)
pgd_params = {'eps': config['eps'],
              'eps_iter': config['eps_iter'],
              'nb_iter': config['nb_iter'],
              'clip_min': 0.,
              'clip_max': 1.,
              'rand_init': True}
# pgd = MadryEtAl(wrap_model, sess=sess)
pgd = MadryEtAl(model, sess=sess)
x_adv = pgd.generate(x, **pgd_params)
y, x_adv = tf.stop_gradient(y), tf.stop_gradient(x_adv)
# logits_adv = wrap_model.get_logits(x_adv)
logits_adv = model.get_logits(x_adv)
loss_adv = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=logits_adv)
loss_adv = tf.reduce_sum(loss_adv)
y_adv = tf.argmax(logits_adv, axis=1)
acc_adv = tf.reduce_sum(
    tf.cast(tf.equal(y_adv, tf.argmax(y, axis=1)), tf.int32))

# Get loss on clean samples
logits_clean = model.get_logits(x)
loss_clean = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=logits_clean)
loss_clean = tf.reduce_sum(loss_clean)
y_pred = tf.argmax(logits_clean, axis=1)
acc_clean = tf.reduce_sum(
    tf.cast(tf.equal(y_pred, tf.argmax(y, axis=1)), tf.int32))
# Since we are not training on clean loss, we stop the gradient
loss_clean = tf.stop_gradient(loss_clean)

# Specify optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_adv)


# Initialize variables
sess.run(tf.global_variables_initializer())
step_per_epoch = n_train // batch_size
start_time = time.time()
sum_loss_clean, sum_loss_adv = 0., 0.
sum_n_clean, sum_n_adv = 0, 0

# Main training loop
for step in range(max_num_training_steps):

    # if step % 100 == 0:
    #     print(step)

    # ind = np.random.randint(len(X_train), size=batch_size)
    # x_batch, y_batch = X_train[ind], y_train[ind]
    # x_adv = pgd.generate_np(x_batch, **pgd_params)
    # feed_dict = {model.input: x_adv, y: y_batch, K.learning_phase(): 1}
    # sess.run(train_step, feed_dict=feed_dict)

    # Get a batch of training samples
    ind = np.random.randint(len(X_train), size=batch_size)
    x_batch, y_batch = X_train[ind], y_train[ind]
    feed_dict = {x: x_batch, y: y_batch}

    # Running one training step
    _, l_clean, l_adv, n_clean, n_adv = sess.run(
        [train_step, loss_clean, loss_adv, acc_clean, acc_adv],
        feed_dict=feed_dict)
    sum_loss_clean += l_clean
    sum_loss_adv += l_adv
    sum_n_clean += n_clean
    sum_n_adv += n_adv

    # For every num_summary_steps, log accuracy and loss
    if step % num_summary_steps == 0:
        log.info("STEP: {}".format(step))
        end_time = time.time()
        log.info("\tElapsed time:\t{:.0f}s".format(end_time - start_time))
        start_time = time.time()
        # loss, acc = model.evaluate(
        #     X_train, np.argmax(y_train, axis=1), verbose=0)
        # loss, acc = eval_cleverhans_clean(X_train, y_train)
        n_samples = float(num_summary_steps*batch_size)
        log.info("\tTrain loss|acc:\t\t{:.2f}|{:.4f}".format(
            sum_loss_clean/n_samples, sum_n_clean/n_samples))
        log.info("\tAdv train loss|acc:\t{:.2f}|{:.4f}".format(
            sum_loss_adv/n_samples, sum_n_adv/n_samples))
        sum_loss_clean, sum_loss_adv = 0., 0.
        sum_n_clean, sum_n_adv = 0, 0
        # loss, acc = model.evaluate(X_val, np.argmax(y_val, axis=1), verbose=0)
        loss, acc = eval_cleverhans_clean(X_val, y_val)
        log.info("\tVal loss|acc:\t\t{:.2f}|{:.4f}".format(loss, acc))

        # X_adv = np.zeros_like(X_val)
        # for i in range(len(X_val) // batch_size):
        #     start = i*batch_size
        #     end = (i + 1)*batch_size
        #     X_adv[start:end] = pgd.generate_np(X_val[start:end], **pgd_params)
        # X_adv[end:] = pgd.generate_np(X_val[end:], **pgd_params)
        # X_adv = pgd.generate_np(X_val, **pgd_params)
        # x_adv = sess.run()
        # loss, acc = model.evaluate(X_adv, np.argmax(y_val, axis=1), verbose=0)
        loss, acc = eval_cleverhans_adv(X_val, y_val)
        log.info("\tAdv val loss|acc:\t{:.2f}|{:.4f}".format(loss, acc))

    # For every num_checkpoint_steps, save weights
    if step % num_checkpoint_steps == 0:
        # loss, acc = model.evaluate(X_val, np.argmax(y_val, axis=1), verbose=0)
        loss, acc = eval_cleverhans_clean(X_val, y_val)
        model_path = "{}{}_step{}_loss{:.2f}_acc{:.4f}.p".format(
            weights_path, model_name, step, loss, acc)
        w = model.get_params()
        weights = sess.run(w)
        with open(model_path, "wb") as f:
            pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
