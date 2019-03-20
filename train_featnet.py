
import os

import pandas as pd
from keras.backend.tensorflow_backend import set_session

from feat_net import FeatNet
from lib.utils import load_gtsrb
from parameters import *
from small_net import gen_balance_data
from stn.conv_model import conv_model_no_color_adjust


def main():

    model_name = 'featnet_hsv_clip'
    use_hsv = True
    learning_rate = 1e-4
    l2_reg = 1e-3
    num_epochs = 10
    batch_size = 128

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    # load dataset
    signnames = pd.read_csv(DATA_DIR + 'signnames.csv')
    X_train, y_train, X_val, y_val, X_test, y_test = load_gtsrb()

    # load stn weights
    model = conv_model_no_color_adjust()
    model.load_weights("./keras_weights/stn_v5.hdf5")
    get_stn_output = K.function([model.layers[0].input, K.learning_phase()],
                                [model.layers[1].output])
    stn_weight = model.layers[1].get_weights()

    # create balance dataset
    X_train_bal, y_train_bal = gen_balance_data(X_train, y_train, [14], r=1)
    y_train_bal = y_train_bal[:, np.newaxis]
    X_val_bal, y_val_bal = gen_balance_data(X_val, y_val, [14], r=1)
    y_val_bal = y_val_bal[:, np.newaxis]
    X_test_bal, y_test_bal = gen_balance_data(X_test, y_test, [14], r=1)
    y_test_bal = y_test_bal[:, np.newaxis]

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
                      load_model=False,
                      stn_weight=stn_weight)
    data = (X_train_bal, y_train_bal, X_val_bal, y_val_bal)
    featnet.train_model(sess, data, dataaug=True,
                        n_epoch=num_epochs, batch_size=batch_size)


if __name__ == '__main__':
    main()
