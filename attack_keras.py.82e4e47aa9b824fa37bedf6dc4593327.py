from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval
from lib.utils import load_gtsrb
from parameters import *
from small_net import *
from stn.conv_model import conv_model_no_color_adjust

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

# Load GTSRB dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_gtsrb()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# Choosing samples to attack
n_attack = 1000
ind_1 = np.where(np.argmax(y_test, axis=1) == 1)
ind_1 = np.where(np.argmax(y_test, axis=1) == 1)

# Obtain Image Parameters
img_rows, img_cols, nchannels = X_train.shape[1:4]
nb_classes = y_train.shape[1]

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

# Load and set up all models
pos_3 = (7, 24, 6, 17)
pos_S = (7, 24, 0, 11)
clf = conv_model_no_color_adjust()
clf.load_weights("./keras_weights/stn_v5.hdf5")
get_stn_output = K.function([clf.layers[0].input, K.learning_phase()],
                            [clf.layers[1].output])
detect_3 = create_simple_cnn(pos_3)
detect_3.load_weights("./keras_weights/1_3.hdf5")
detect_S = create_simple_cnn(pos_S)
detect_S.load_weights("./keras_weights/14_S.hdf5")

wrap_clf = KerasModelWrapper(clf)
preds = clf(x)

eval_par = {'batch_size': 128}
acc = model_eval(sess, x, y, preds, X_test, y_test, args=eval_par)
print('Test accuracy on legitimate test examples: {0}'.format(acc))
report.clean_train_clean_eval = acc

# fgsm = FastGradientMethod(wrap_clf, sess=sess)
# fgsm_params = {'eps': 0.1,
#                'clip_min': 0.,
#                'clip_max': 1.}
# adv_x = fgsm.generate(x, **fgsm_params)
# # Consider the attack to be constant
# adv_x = tf.stop_gradient(adv_x)
# preds_adv = clf(adv_x)

# # Evaluate the accuracy of the MNIST model on adversarial examples
# acc = model_eval(sess, x, y, preds_adv, X_test, y_test, args=eval_par)
# print('Test accuracy on adversarial examples: %0.4f\n' % acc)

# CarliniWagner attack
# cw = CarliniWagnerL2(clf, back='tf', sess=sess)
from lib.white_box_attack import CarliniWagnerL2_WB

ensemble = []
for i in range(nb_classes):
    nets = []
    if i == 1:
        nets.append(KerasModelWrapper(detect_3))
    elif i == 14:
        nets.append(KerasModelWrapper(detect_S))
    ensemble.append(nets)
print(ensemble)

cw = CarliniWagnerL2_WB(clf, ensemble=ensemble, back='tf', sess=sess)

# Untargeted attack
attack_iterations = 100
# adv_inputs = X_test[:n_attack]
# adv_ys = None
# yname = 'y'
cw_params = {'binary_search_steps': 1,
             'max_iterations': attack_iterations,
             'learning_rate': 0.1,
             'batch_size': 1000,
             'initial_const': 1e-2}
# adv_x = cw.generate(x, **cw_params)
# preds_adv = clf(adv_x)
# acc = model_eval(sess, x, y, preds_adv, X_test[:n_attack],
#                  y_test[:n_attack], args=eval_par)
# print('Test accuracy on CW adversarial examples: %0.4f\n' % acc)
adv = cw.generate_np(X_test[:n_attack], **cw_params)
print(adv)

# batch_size = 1,
# confidence = 0,
# learning_rate = 5e-3,
# binary_search_steps = 5,
# max_iterations = 1000,
# abort_early = True,
# initial_const = 1e-2,
# clip_min = 0,
# clip_max = 1
