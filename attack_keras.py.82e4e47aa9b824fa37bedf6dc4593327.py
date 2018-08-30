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
pos_2 = (7, 23, 5, 17)
pos_0 = (7, 24, 15, 27)  # Good for class 0 - 5
pos_3 = (7, 24, 6, 17)

pos_S = (7, 24, 0, 11)
pos_T = (7, 23, 8, 17)
pos_O = (6, 23, 13, 24)
pos_P = (6, 23, 22, 32)

clf = conv_model_no_color_adjust()
clf.load_weights("./keras_weights/stn_v5.hdf5")

stn_weight = clf.layers[1].get_weights()
detect_3 = create_simple_cnn(pos_3)
detect_3.load_weights("./keras_weights/1_3.hdf5")
detect_0 = create_simple_cnn(pos_0)
detect_0.load_weights("./keras_weights/0.hdf5")
detect_S = create_simple_cnn(pos_S)
detect_S.load_weights("./keras_weights/14_S.hdf5")
detect_T = create_simple_cnn(pos_T)
detect_T.load_weights("./keras_weights/14_T.hdf5")
detect_O = create_simple_cnn(pos_O)
detect_O.load_weights("./keras_weights/14_O.hdf5")
detect_P = create_simple_cnn(pos_P)
detect_P.load_weights("./keras_weights/14_P.hdf5")

check_cnn = True
if check_cnn:
    y_test_cat = np.argmax(y_test, axis=1)
    print(eval_simple_cnn(detect_3, [1], X_test, y_test_cat))
    print(eval_simple_cnn(detect_0, [0, 1, 2, 3, 4, 5], X_test, y_test_cat))
    print(eval_simple_cnn(detect_S, [14], X_test, y_test_cat))
    print(eval_simple_cnn(detect_T, [14], X_test, y_test_cat))
    print(eval_simple_cnn(detect_O, [14], X_test, y_test_cat))
    print(eval_simple_cnn(detect_P, [14], X_test, y_test_cat))

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
from lib.white_box_attack import CarliniWagnerL2_WB

ensemble = []
for i in range(nb_classes):
    nets = []
    if i == 1:
        nets.append(detect_3)
        nets.append(detect_0)
    elif i == 14:
        nets.append(detect_S)
        nets.append(detect_T)
        nets.append(detect_O)
        nets.append(detect_P)
    ensemble.append(nets)

wrap_ensemble = [[KerasModelWrapper(net) for net in nets] for nets in ensemble]
print(ensemble)

# Set up CW attack params
attack_iterations = 200
cw_params = {'binary_search_steps': 1,
             'max_iterations': attack_iterations,
             'learning_rate': 0.1,
             'batch_size': n_attack,
             'initial_const': 15,
             'y_target': y_target}
# cw_params = {'binary_search_steps': 1,
#              'max_iterations': attack_iterations,
#              'learning_rate': 0.1,
#              'batch_size': n_attack,
#              'initial_const': 1e-2}

# cw = CarliniWagnerL2(wrap_clf, back='tf', sess=sess)
# adv_x = cw.generate(x, **cw_params)
# preds_adv = clf(adv_x)
# acc = model_eval(sess, x, y, preds_adv, X_test[:n_attack],
#                  y_test[:n_attack], args=eval_par)
# print('Test accuracy on CW adversarial examples: %0.4f\n' % acc)

# ======================= MY CODE ======================= #
cw = CarliniWagnerL2_WB(wrap_clf, ensemble=wrap_ensemble, back='tf', sess=sess)
adv = cw.generate_np(X_atk, **cw_params)
# print(adv)

# Evaluate clean samples
pred_clf = clf.predict(X_atk)
pred_3 = detect_3.predict(X_atk)
pred_S = detect_S.predict(X_atk)
n_correct_clf = 0
n_correct_3 = 0
n_correct_S = 0
for i in range(n_attack):
    if i < n_attack//2:
        if np.argmax(pred_clf[i]) == 1:
            n_correct_clf += 1
            if np.argmax(pred_3[i]) == 1:
                n_correct_3 += 1
    else:
        if np.argmax(pred_clf[i]) == 14:
            n_correct_clf += 1
            if np.argmax(pred_S[i]) == 1:
                n_correct_S += 1
print("Correct classification only by clf: ", n_correct_clf)
print("Correct classification by clf & detect_3: ", n_correct_3)
print("Correct classification by clf & detect_S: ", n_correct_S)

# Evaluate adv
pred_clf = np.argmax(clf.predict(adv), axis=1)
pred_ensemble = []
for i in range(nb_classes):
    pred_nets = []
    for net in ensemble[i]:
        pred_nets.append(np.argmax(net.predict(adv), axis=1))
    pred_ensemble.append(pred_nets)

n_suc_clf = 0
n_suc_1 = 0
n_suc_14 = 0
for i in range(n_attack):
    if i < n_attack//2:
        if pred_clf[i] == 14:
            n_suc_clf += 1
            pred_ens_x = []
            for pred_net in pred_ensemble[14]:
                pred_ens_x.append(pred_net[i])
            if 0 not in pred_ens_x:
                n_suc_14 += 1
            print(pred_ens_x)
    else:
        if pred_clf[i] == 1:
            n_suc_clf += 1
            pred_ens_x = []
            for pred_net in pred_ensemble[1]:
                pred_ens_x.append(pred_net[i])
            if 0 not in pred_ens_x:
                n_suc_1 += 1
# print(pred_ensemble)

# Eval untargeted
# for i in range(n_attack):
#     if i < n_attack//2:
#         if np.argmax(pred_clf[i]) != 1:
#             n_suc_clf += 1
#             if np.argmax(pred_3[i]) != 1:
#                 n_suc_3 += 1
#     else:
#         if np.argmax(pred_clf[i]) != 14:
#             n_suc_clf += 1
#             if np.argmax(pred_S[i]) != 1:
#                 n_suc_S += 1

print("\n\n==========================================\n\n")
print("Successful attack only on clf: ", n_suc_clf)
print("Successful attack on clf & detect_3: ", n_suc_1)
print("Successful attack on clf & detect_S: ", n_suc_14)

# batch_size = 1,
# confidence = 0,
# learning_rate = 5e-3,
# binary_search_steps = 5,
# max_iterations = 1000,
# abort_early = True,
# initial_const = 1e-2,
# clip_min = 0,
# clip_max = 1
