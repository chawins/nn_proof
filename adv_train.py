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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
reg = config['reg']

# Set TF random seed to improve reproducibility
tf.set_random_seed(config['random_seed'])

if K.image_dim_ordering() != 'tf':
    K.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

# Create TF session and set as Keras backend session
config_sess = tf.ConfigProto()
config_sess.gpu_options.allow_growth = True
sess = tf.Session(config=config_sess)
K.set_session(sess)

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

    from adv_model import *
    # model = CnnGtsrbV1('CnnGtsrbV1', nb_classes)
    model = CnnGtsrbV2(model_name, nb_classes)

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

    # Build MNIST model
    model = build_mnist()

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

else:
    raise NotImplementedError()

# Set up adversarial example generation using cleverhans
pgd_params = {'eps': config['eps'],
              'eps_iter': config['eps_iter'],
              'nb_iter': config['nb_iter'],
              'clip_min': 0.,
              'clip_max': 1.,
              'rand_init': True}
pgd = MadryEtAl(model, sess=sess)
x_adv = pgd.generate(x, **pgd_params)
# Stop gradient on unnecessary tensors, save lots of memory
y, x_adv = tf.stop_gradient(y), tf.stop_gradient(x_adv)
# Set up loss and accuracy for adversarial training
logits_adv = model.get_logits(x_adv)
loss_adv = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=logits_adv)
loss_adv = tf.reduce_sum(loss_adv)
y_adv = tf.argmax(logits_adv, axis=1)
acc_adv = tf.reduce_sum(
    tf.cast(tf.equal(y_adv, tf.argmax(y, axis=1)), tf.int32))

# Get loss and accuracy on clean samples
logits_clean = model.get_logits(x)
loss_clean = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=logits_clean)
loss_clean = tf.reduce_sum(loss_clean)
y_pred = tf.argmax(logits_clean, axis=1)
acc_clean = tf.reduce_sum(
    tf.cast(tf.equal(y_pred, tf.argmax(y, axis=1)), tf.int32))
# Since we are not training on clean loss, we stop the gradient
loss_clean = tf.stop_gradient(loss_clean)

# Add regularization loss
loss_reg = 0
for w in tf.trainable_variables(scope=model_name):
    if len(w.shape) != 1:
        loss_reg += reg*tf.reduce_sum(tf.square(w))
loss_adv += loss_reg

# Add regularization loss
# loss_adv += tf.losses.get_regularization_loss()

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
        # Log step and time
        log.info("STEP: {}".format(step))
        end_time = time.time()
        log.info("\tElapsed time:\t{:.0f}s".format(end_time - start_time))
        start_time = time.time()
        n_samples = float(num_summary_steps*batch_size)

        # Log accuracy and loss on training set
        log.info("\tTrain loss|acc:\t\t{:.2f}|{:.4f}".format(
            sum_loss_clean/n_samples, sum_n_clean/n_samples))
        log.info("\tAdv train loss|acc:\t{:.2f}|{:.4f}".format(
            sum_loss_adv/n_samples, sum_n_adv/n_samples))
        sum_loss_clean, sum_loss_adv = 0., 0.
        sum_n_clean, sum_n_adv = 0, 0

        # Log accuracy and loss on validation set
        loss, acc = eval_cleverhans_clean(X_val, y_val)
        log.info("\tVal loss|acc:\t\t{:.2f}|{:.4f}".format(loss, acc))
        loss, acc = eval_cleverhans_adv(X_val, y_val)
        log.info("\tAdv val loss|acc:\t{:.2f}|{:.4f}".format(loss, acc))

    # For every num_checkpoint_steps, save weights
    if step % num_checkpoint_steps == 0:
        loss, acc = eval_cleverhans_clean(X_val, y_val)
        model_path = "{}{}_step{}_loss{:.2f}_acc{:.4f}.p".format(
            weights_path, model_name, step, loss, acc)
        w = model.get_params()
        weights = sess.run(w)
        with open(model_path, "wb") as f:
            pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
