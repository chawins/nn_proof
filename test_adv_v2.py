import json
import logging
import os
import time
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


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


# def build_gtsrb():

#     from keras.layers import (Activation, Dense, Dropout, Flatten, Lambda,
#                               MaxPooling2D, Reshape)

#     model = Sequential()
#     model.add(Reshape((32, 32, 3), input_shape=(32, 32, 3)))
#     model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dense(43, activation='softmax'))

#     adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer=adam, metrics=['accuracy'])

#     return model


def build_gtsrb():

    from keras.layers import (Activation, Dense, Dropout, Flatten, Lambda,
                              MaxPooling2D, Reshape)

    model = Sequential()
    # model.add(Lambda(
    #     lambda x: x*2 - 1.,
    #     input_shape=(32, 32, 3),
    #     output_shape=(32, 32, 3)))
    # model.add(Conv2D(16, (5, 5), padding='same',
    #                  activation='relu'))
    model.add(Reshape((32, 32, 3), input_shape=(32, 32, 3)))
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(96, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(192, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Flatten())
    # model.add(Dropout(0.6))
    model.add(Dense(43, activation='softmax'))

    # adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # model.compile(loss='sparse_categorical_crossentropy',
    #               optimizer=adam, metrics=['accuracy'])

    return model


# Object used to keep track of (and return) key accuracies
# report = AccuracyReport()

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

if K.image_dim_ordering() != 'tf':
    K.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

# Create TF session and set as Keras backend session
config_sess = tf.ConfigProto()
config_sess.gpu_options.allow_growth = True
sess = tf.Session(config=config_sess)
# sess = tf.Session()
K.set_session(sess)

# Get TF logger
log = logging.getLogger('test_adv')
log.setLevel(logging.DEBUG)
# Create formatter and add it to the handlers
formatter = logging.Formatter(
    '[%(levelname)s %(asctime)s %(name)s] %(message)s')
# Create file handler
fh = logging.FileHandler('test_adv.log', mode='w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

# set_log_level(logging.DEBUG)

with open('config.json') as config_file:
    config = json.load(config_file)
dataset = config['dataset']
model_path = config['model_path']
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
batch_size = config['training_batch_size']

if dataset == 'gtsrb':
    # Load GTSRB dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_gtsrb()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    # Obtain Image Parameters
    n_train, img_rows, img_cols, nchannels = X_train.shape
    nb_classes = y_train.shape[1]

    model = build_gtsrb()
    # model = build_cnn_no_stn()
    # model = conv_model_no_color_adjust()
    # from adv_model import CnnGtsrb
    # with tf.device("/gpu:1"):
    #     model = CnnGtsrb('model1', nb_classes)

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
    x = tf.placeholder(tf.float32, shape=(batch_size, img_rows, img_cols))
    y = tf.placeholder(tf.float32, shape=(batch_size, nb_classes))

# Set up adversarial example generation using cleverhans
wrap_model = KerasModelWrapper(model)
pgd_params = {'eps': 0.3,
              'eps_iter': 0.01,
              'nb_iter': 40,
              'clip_min': 0.,
              'clip_max': 1.,
              'rand_init': True}
pgd = MadryEtAl(wrap_model, sess=sess)
# pgd = MadryEtAl(model, sess=sess)

# logits = model.layers[-2].output
# loss = tf.nn.softmax_cross_entropy_with_logits_v2(
#     labels=y, logits=logits)

# train_step = tf.nn.softmax_cross_entropy_with_logits_v2(
#     labels=y, logits=wrap_model.get_logits(x))
x_adv = pgd.generate(x, **pgd_params)
logits_adv = wrap_model.get_logits(x_adv)
# logits_adv = model.get_logits(x_adv)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=logits_adv)

# train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])

# Initialize variables
sess.run(tf.global_variables_initializer())
step_per_epoch = n_train // batch_size
start_time = time.time()

from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
with tf.device('/gpu:0'):  # Replace with device you are interested in
    bytes_in_use = BytesInUse()
print(sess.run(bytes_in_use))
with tf.device('/gpu:1'):  # Replace with device you are interested in
    bytes_in_use = BytesInUse()
print(sess.run(bytes_in_use))

model.fit(X_train, y_train)

# Main training loop
for step in range(max_num_training_steps):

    if step % 100 == 0:
        print(step)

    # ind = np.random.randint(len(X_train), size=batch_size)
    # x_batch, y_batch = X_train[ind], y_train[ind]
    # x_adv = pgd.generate_np(x_batch, **pgd_params)
    # feed_dict = {model.input: x_adv, y: y_batch, K.learning_phase(): 1}
    # sess.run(train_step, feed_dict=feed_dict)

    # Get a batch of training samples
    ind = np.random.randint(len(X_train), size=batch_size)
    x_batch, y_batch = X_train[ind], y_train[ind]
    feed_dict = {x: x_batch, y: y_batch, K.learning_phase(): 1}

    # Running one training step
    sess.run(train_step, feed_dict=feed_dict)

    # For every num_summary_steps, log accuracy and loss
    if step % num_summary_steps == 0:
        log.info("Epoch: {}".format(step // step_per_epoch))
        end_time = time.time()
        log.info("\tTime per epoch: {:.0f}s".format(end_time - start_time))
        start_time = time.time()
        loss, acc = model.evaluate(
            X_train, np.argmax(y_train, axis=1), verbose=0)
        log.info("\tTrain loss|acc: {:.2f}|{:.4f}".format(loss, acc))
        loss, acc = model.evaluate(X_val, np.argmax(y_val, axis=1), verbose=0)
        log.info("\tVal loss|acc: {:.2f}|{:.4f}".format(loss, acc))

        x_adv = pgd.generate_np(X_val, **pgd_params)
        loss, acc = model.evaluate(x_adv, np.argmax(y_val, axis=1), verbose=0)
        log.info("\tAdversarial val loss|acc: {:.2f}|{:.4f}".format(loss, acc))

    # For every num_checkpoint_steps, save weights
    if step % num_checkpoint_steps == 0:
        loss, acc = model.evaluate(X_val, np.argmax(y_val, axis=1), verbose=0)
        model.save("{}_step{}_loss{:.2f}_acc{:.4f}.h5".format(
            model_path, step, loss, acc))
