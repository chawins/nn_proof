
import logging
import os

import pandas as pd
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils import set_log_level
from cleverhans.utils_keras import KerasModelWrapper
from keras.backend.tensorflow_backend import set_session

from feat_net import FeatNet
from lib.custom_cw import CustomCarliniWagnerL2
from lib.utils import load_gtsrb
from parameters import *
from small_net import gen_balance_data
from stn.conv_model import conv_model_no_color_adjust


def evaluate(model, featnet, x, sess):
    y_model = model.predict(x).argmax(-1)
    y_featnet = featnet.predict_model(sess, x).squeeze()
    return y_model, y_featnet


def main():

    model_name = 'featnet_hsv_clip'
    use_hsv = True
    batch_size = 100
    bin_search_steps = 10
    attack_iterations = 500
    num_attacks = 100
    thres = 3.1
    learning_rate = 1e-4
    l2_reg = 1e-3

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    # load dataset
    _, _, _, _, X_test, y_test = load_gtsrb()

    # load stn weights
    model = conv_model_no_color_adjust()
    model.load_weights("./keras_weights/stn_v5.hdf5")
    stn_weight = model.layers[1].get_weights()
    wrap_clf = KerasModelWrapper(model)

    # predefine position of patches to crop
    pos_S = (7, 24, 0, 11)
    pos_T = (7, 23, 8, 17)
    pos_O = (6, 23, 13, 24)
    pos_P = (6, 23, 22, 32)

    featnet = FeatNet(model_name, [32, 32, 3], [1],
                      [pos_S, pos_T, pos_O, pos_P],
                      hsv=use_hsv,
                      learning_rate=learning_rate,
                      reg=l2_reg,
                      save_path="model/%s.h5" % model_name,
                      load_model=True,
                      stn_weight=stn_weight)

    keras.backend.set_learning_phase(0)
    set_log_level(logging.INFO)

    def attack_wrapper(x, y_target=None):

        x_adv = np.zeros_like(x)
        x1 = x[:(len(x) // batch_size) * batch_size]
        x2 = x[len(x1):]

        cw_params = {'binary_search_steps': bin_search_steps,
                     'max_iterations': attack_iterations,
                     'learning_rate': 0.01,
                     'batch_size': batch_size,
                     'initial_const': 10}
        if y_target is not None:
            cw_params['y_target'] = y_target[:len(x1)]
        cw = CustomCarliniWagnerL2(wrap_clf, featnet, thres=thres, sess=sess)
        # cw = CarliniWagnerL2(wrap_clf, sess=sess)
        x_adv[:len(x1)] = cw.generate_np(x1, **cw_params)

        cw_params['batch_size'] = len(x2)
        if y_target is not None:
            cw_params['y_target'] = y_target[len(x1):]
        x_adv[len(x1):] = cw.generate_np(x2, **cw_params)

        return x_adv

    X_atk = X_test[y_test != 14]
    y_target = np.zeros((len(X_atk), )) + 14
    y_target = to_categorical(y_target, 43)
    X_adv = attack_wrapper(X_atk, y_target=y_target)

    y_test_model, y_test_fn = evaluate(model, featnet, X_test, sess)
    y_adv_model, y_adv_fn = evaluate(model, featnet, X_adv, sess)

    test_fp = np.mean(
        (y_test_model[y_test != 14] == 14) & (y_test_fn[y_test != 14] >= thres))
    print('Test FP rate: %.4f' % test_fp)
    test_fn = np.mean(
        (y_test_model[y_test == 14] != 14) | (y_test_fn[y_test == 14] < thres))
    print('Test FN rate: %.4f' % test_fn)
    adv_fp = np.mean((y_adv_model == 14) & (y_adv_fn >= thres))
    print('Adv FP rate: %.4f' % adv_fp)
    print('Adv model FP rate: %.4f' % np.mean(y_adv_model == 14))
    print('Adv featnet FP rate: %.4f' % np.mean(y_adv_fn >= thres))

    ind = (y_adv_model == 14) & (y_adv_fn >= thres)
    # ind = (y_adv_model == 14)
    l2_dist = np.mean(np.sqrt(np.sum((X_atk[ind] - X_adv[ind])**2, (1, 2, 3))))
    linf_dist = np.mean(np.max(np.abs(X_atk[ind] - X_adv[ind]), (1, 2, 3)))
    print('Mean successful l-2 dist: %.4f' % l2_dist)
    print('Mean successful l-inf dist: %.4f' % linf_dist)

    # turn STOP to non-STOP
    # X_atk = X_test[y_test == 14]
    # X_adv = attack_wrapper(X_atk)
    # y_test_model, y_test_fn = evaluate(model, featnet, X_test, sess)
    # y_adv_model, y_adv_fn = evaluate(model, featnet, X_adv, sess)
    #
    # test_fn = np.mean((y_test_model != 14) & (y_test_fn < thres))
    # print('Test FN rate: %.4f' % test_fn)
    # adv_fn = np.mean((y_adv_model != 14) & (y_adv_fn < thres))
    # print('Adv FN rate: %.4f' % adv_fn)
    # print('Adv model FN rate: %.4f' % np.mean(y_adv_model != 14))
    # print('Adv featnet FN rate: %.4f' % np.mean(y_adv_fn < thres))
    #
    # # ind = (y_adv_model != 14) & (y_adv_fn < thres)
    # ind = (y_adv_model != 14)
    # l2_dist = np.mean(np.sqrt(np.sum((X_atk[ind] - X_adv[ind])**2, (1, 2, 3))))
    # linf_dist = np.mean(np.max(np.abs(X_atk[ind] - X_adv[ind]), (1, 2, 3)))
    # print('Mean successful l-2 dist: %.4f' % l2_dist)
    # print('Mean successful l-inf dist: %.4f' % linf_dist)


if __name__ == '__main__':
    main()
