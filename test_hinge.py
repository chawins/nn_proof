from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib
from skimage.transform import resize
from imageio import imread, imwrite
import pandas as pd
import glob
import pickle
import keras.backend as K

from stn.conv_model import conv_model
from stn.conv_model import conv_model_no_color_adjust
from sklearn.utils import resample
from lib.utils import load_gtsrb
from keras.metrics import sparse_categorical_accuracy

from parameters import *
from small_net import *

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse


def main(model_name):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras.backend.tensorflow_backend import set_session
    set_session(sess)

    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    n_train = len(X_train)
    n_val = int(n_train*0.1)
    ind = np.arange(n_train)
    np.random.shuffle(ind)
    X_val, y_val = X_train[ind[:n_val]], y_train[ind[:n_val]]
    X_train, y_train = X_train[ind[n_val:]], y_train[ind[n_val:]]

    data = (X_train[:, :, :, np.newaxis], y_train, 
            X_val[:, :, :, np.newaxis], y_val)

    from cleverhans.utils import set_log_level
    import logging

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    # model_name = "softmax_xent"

    from hinge_net import HingeNet

    hingenet = HingeNet(model_name, [28, 28, 1], [10], learning_rate=1e-4, 
                        load_model=False, save_path="model/" + model_name + ".h5")
    hingenet.train_model(sess, data, n_epoch=10, batch_size=128)

    print("Weights")
    weights = []
    w = hingenet.model.get_weights()
    for i, w in enumerate(w):
        if i % 2 == 0:
            weight = [np.sum(np.square(w)), np.sum(np.abs(w)), np.mean(w), np.std(w)]
            print(weight)

    print("Clean accuracy")
    clean_acc = hingenet.eval_model(sess, (X_test[:, :, :, np.newaxis], np.argmax(y_test, axis=1)))
    print(clean_acc)

    # Least likeley class
    n_attack = 1000
    X_atk = X_test[:, :, :, np.newaxis][:n_attack]
    y_atk = np.argmax(y_test[:n_attack], axis=1)

    y_pred = hingenet.predict_model(sess, X_atk)
    y_target = to_categorical(np.argmin(y_pred, axis=1), 10)

    print("Gradients")
    grads = 0
    for i in range(int(n_attack/100)):
        grad = sess.run(hingenet.local_grad, 
                        feed_dict={hingenet.x: X_atk[i*100:(i + 1)*100],
                                hingenet.y: np.argmax(y_target[i*100:(i + 1)*100], axis=1)})
        grads += np.sum(np.square(grad))
    avg_grad = grads/n_attack
    print(avg_grad)

    print("Accuracy on X_atk")
    print(hingenet.eval_model(sess, (X_atk, y_atk)))

    keras.backend.set_learning_phase(0)
    set_log_level(logging.INFO)

    # CarliniWagner attack
    from lib.my_cw import CarliniWagnerL2
    attack_iterations = 200
    cw_params = {'binary_search_steps': 3,
                'max_iterations': attack_iterations,
                'learning_rate': 0.1,
                'batch_size': 100,
                'initial_const': 10,
                'y_target': y_target}
    cw = CarliniWagnerL2(hingenet, sess=sess)
    adv = cw.generate_np(X_atk, **cw_params)

    print("Attack done")
    adv_acc = hingenet.eval_model(sess, (adv, np.argmax(y_target, axis=1)))
    print(adv_acc)
    output = hingenet.predict_model(sess, adv)
    y_pred = np.argmax(output, axis=1)
    max_out = np.max(output, axis=1)
    ind = []
    y_target = np.argmax(y_target, axis=1)
    for i in range(n_attack):
        if y_pred[i] == y_target[i]:
            ind.append(i)
    ind = np.array(ind)
    avg_out = np.mean(max_out[ind])
    dist = np.mean(np.sqrt(np.sum((adv[ind] - X_atk[ind])**2, (1, 2, 3))))
    print(avg_out)
    print(dist)

    try:
        out = pickle.load(open(model_name + ".p", "rb"))
        out["Clean acc"].append(clean_acc[0])
        out["Mean norm grad"].append(avg_grad)
        out["Adv suc"].append(adv_acc[0])
        out["Mean adv score"].append(avg_out)
        out["Mean suc dist"].append(dist)
    except FileNotFoundError:
        out = {"Clean acc": [clean_acc[0]],
               "Mean norm grad": [avg_grad],
               "Adv suc": [adv_acc[0]],
               "Mean adv score": [avg_out],
               "Mean suc dist": [dist]}
    # pickle_dump = [clean_acc, avg_grad, adv_acc, avg_out, dist]
    pickle.dump(out, open(model_name + ".p", "wb"))


def write(model_name):
    out = pickle.load(open(model_name + ".p", "rb"))
    with open(model_name + ".txt", "w") as myfile:
        myfile.write(model_name + "\n")
        myfile.write("Clean acc: {:.3g}, {:.3g}, {:.3g}, {:.3g}, {:.3g}\n".format(*out["Clean acc"]))
        myfile.write("Mean norm grad: {:.3g}, {:.3g}, {:.3g}, {:.3g}, {:.3g}\n".format(*out["Mean norm grad"]))
        myfile.write("Adv suc: {:.3g}, {:.3g}, {:.3g}, {:.3g}, {:.3g}\n".format(*out["Adv suc"]))
        myfile.write("Mean adv score: {:.3g}, {:.3g}, {:.3g}, {:.3g}, {:.3g}\n".format(*out["Mean adv score"]))
        myfile.write("Mean suc dist: {:.3g}, {:.3g}, {:.3g}, {:.3g}, {:.3g}\n".format(*out["Mean suc dist"]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='Experiment/model name')
    parser.add_argument("--write", help="Write outputs", action="store_true")
    args = parser.parse_args()
    if args.write:
        write(args.model_name)
    else:
        main(args.model_name)

