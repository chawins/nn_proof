from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval
from lib.utils import load_gtsrb
from parameters import *
from small_net import *
from stn.conv_model import conv_model_no_color_adjust
import logging

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

keras.layers.core.K.set_learning_phase(0)

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

if K.image_dim_ordering() != 'tf':
    K.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

# Create TF session and set as Keras backend session
sess = tf.Session()
K.set_session(sess)

set_log_level(logging.DEBUG)

# Load GTSRB dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_gtsrb()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# Obtain Image Parameters
img_rows, img_cols, nchannels = X_train.shape[1:4]
nb_classes = y_train.shape[1]

# Choosing samples to attack
n_attack = 500
ind_1 = np.where(np.argmax(y_test, axis=1) == 1)
ind_14 = np.where(np.argmax(y_test, axis=1) == 14)
X_atk = np.zeros((n_attack, ) + X_test.shape[1:4])
X_atk[:n_attack//2] = X_test[ind_1][:n_attack//2]
X_atk[n_attack//2:] = X_test[ind_14][:n_attack//2]

y_target = np.zeros((n_attack, ))
y_target[:n_attack//2] = 14
y_target[n_attack//2:] = 1
y_target = to_categorical(y_target, nb_classes)
# print(y_target)

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

# Load and set up all models
from stn.conv_model import build_cnn_no_stn
clf = build_cnn_no_stn()
clf.load_weights("./keras_weights/cnn_v4.hdf5")
# clf = conv_model_no_color_adjust()
# clf.load_weights("./keras_weights/stn_v5.hdf5")

wrap_clf = KerasModelWrapper(clf)
preds = clf(x)

eval_par = {'batch_size': 128}
acc = model_eval(sess, x, y, preds, X_test, y_test, args=eval_par)
print('Test accuracy on legitimate test examples: {0}'.format(acc))
report.clean_train_clean_eval = acc

# FGSM
# fgsm = FastGradientMethod(wrap_clf, sess=sess)
# fgsm_params = {'eps': 0.1,
#                'clip_min': 0.,
#                'clip_max': 1.}
# adv_x = fgsm.generate_np(X_atk, **fgsm_params)

# # Evaluate the accuracy of the MNIST model on adversarial examples
# acc = model_eval(sess, x, y, preds_adv, X_test, y_test, args=eval_par)
# print('Test accuracy on adversarial examples: %0.4f\n' % acc)

# CarliniWagner attack
# attack_iterations = 200
# cw_params = {'binary_search_steps': 3,
#              'max_iterations': attack_iterations,
#              'learning_rate': 0.1,
#              'batch_size': n_attack,
#              'initial_const': 10,
#              'y_target': y_target}
# # cw_params = {'binary_search_steps': 1,
# #              'max_iterations': attack_iterations,
# #              'learning_rate': 0.1,
# #              'batch_size': n_attack,
# #              'initial_const': 10}
# cw = CarliniWagnerL2(wrap_clf, back='tf', sess=sess)
# adv = cw.generate_np(X_atk, **cw_params)

from cleverhans.attacks import MadryEtAl
pgd_params = {'eps': 0.3,
              'eps_iter': 0.01,
              'nb_iter': 40,
              'clip_min': 0.,
              'clip_max': 1.,
              'rand_init': True}
pgd = MadryEtAl(wrap_clf, sess=sess)
adv = pgd.generate_np(X_atk, **pgd_params)

# adv_x = cw.generate(x, **cw_params)
# preds_adv = clf(adv_x)
# acc = model_eval(sess, x, y, preds_adv, X_test[:n_attack],
#                  y_test[:n_attack], args={'batch_size': n_attack})
# print('Test accuracy on CW adversarial examples: %0.4f\n' % acc)

pred = clf.predict(adv)
# print(np.sum(np.argmax(pred, axis=1) != np.argmax(y_test[:n_attack], axis=1)))
# pred_orig = clf.predict(X_atk)
# print(np.sum(np.argmax(pred, axis=1) != np.argmax(pred_orig, axis=1)))
print(np.sum(np.argmax(pred, axis=1) == np.argmax(y_target, axis=1)))

# Save some images
import scipy.misc

# DIR = './vis/exp2/'
# for i in range(20):
#     scipy.misc.imsave('{}org_{}.png'.format(DIR, i), X_atk[i])
#     scipy.misc.imsave('{}adv_{}.png'.format(DIR, i), adv[i])
#     scipy.misc.imsave('{}org_{}.png'.format(
#         DIR, n_attack//2 + i), X_atk[n_attack//2 + i])
#     scipy.misc.imsave('{}adv_{}.png'.format(
#         DIR, n_attack//2 + i), adv[n_attack//2 + i])
