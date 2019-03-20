import keras
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Dense, Dropout, Flatten, Lambda,
                          MaxPooling2D, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

from stn.spatial_transformer import SpatialTransformer


def locnet():
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((64, 6), dtype='float32')
    weights = [W, b.flatten()]
    locnet = Sequential()

    locnet.add(Conv2D(16, (7, 7), padding='valid', input_shape=(32, 32, 3)))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Conv2D(32, (5, 5), padding='valid'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Conv2D(64, (3, 3), padding='valid'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))

    locnet.add(Flatten())
    locnet.add(Dense(128))
    locnet.add(Activation('elu'))
    locnet.add(Dense(64))
    locnet.add(Activation('elu'))
    locnet.add(Dense(6, weights=weights))

    return locnet


def locnet_v2():

    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((1024, 6), dtype='float32')
    weights = [W, b.flatten()]

    # Regularization
    l2_reg = keras.regularizers.l2(1e-4)

    # Build model
    inpt = keras.layers.Input(shape=(32, 32, 3))
    conv1 = keras.layers.Convolution2D(
        32, (5, 5), padding='same', activation='relu')(inpt)
    drop1 = keras.layers.Dropout(rate=0.1)(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = keras.layers.Convolution2D(
        64, (5, 5), padding='same', activation='relu')(pool1)
    drop2 = keras.layers.Dropout(rate=0.2)(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = keras.layers.Convolution2D(
        128, (5, 5), padding='same', activation='relu')(pool2)
    drop3 = keras.layers.Dropout(rate=0.3)(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)

    pool4 = keras.layers.MaxPooling2D(pool_size=(4, 4))(pool1)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(pool2)

    flat1 = keras.layers.Flatten()(pool4)
    flat2 = keras.layers.Flatten()(pool5)
    flat3 = keras.layers.Flatten()(pool3)

    merge = keras.layers.Concatenate(axis=-1)([flat1, flat2, flat3])
    dense1 = keras.layers.Dense(1024, activation='relu',
                                kernel_regularizer=l2_reg)(merge)
    drop4 = keras.layers.Dropout(rate=0.5)(dense1)
    output = keras.layers.Dense(
        6, activation=None, kernel_regularizer=l2_reg, weights=weights)(drop4)
    locnet = keras.models.Model(inputs=inpt, outputs=output)

    return locnet


def locnet_v3():

    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((128, 6), dtype='float32')
    weights = [W, b.flatten()]

    # Regularization
    l2_reg = keras.regularizers.l2(0)

    # Build model
    inpt = keras.layers.Input(shape=(32, 32, 3))
    dim_reduce = keras.layers.Convolution2D(
        1, (1, 1), padding='same', activation='relu')(inpt)
    conv1 = keras.layers.Convolution2D(
        16, (5, 5), padding='same', activation='relu')(dim_reduce)
    # conv1 = keras.layers.Convolution2D(
    #     16, (5, 5), padding='same', activation='relu')(inpt)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = keras.layers.Convolution2D(
        32, (5, 5), padding='same', activation='relu')(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = keras.layers.Convolution2D(
        64, (5, 5), padding='same', activation='relu')(pool2)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    pool4 = keras.layers.MaxPooling2D(pool_size=(4, 4))(pool1)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(pool2)

    flat1 = keras.layers.Flatten()(pool4)
    flat2 = keras.layers.Flatten()(pool5)
    flat3 = keras.layers.Flatten()(pool3)

    merge = keras.layers.Concatenate(axis=-1)([flat1, flat2, flat3])
    dense1 = keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=l2_reg)(merge)
    output = keras.layers.Dense(
        6, activation=None, kernel_regularizer=l2_reg, weights=weights)(dense1)
    locnet = keras.models.Model(inputs=inpt, outputs=output)

    return locnet


def conv_model(input_shape=(32, 32, 3)):

    l2_reg = 0.05

    model = Sequential()
    model.add(Lambda(
        lambda x: x * 2 - 1.,
        input_shape=(32, 32, 3),
        output_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(10, (1, 1), padding='same',
                     kernel_regularizer=l2(l2_reg)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(Conv2D(3, (1, 1), padding='same', kernel_regularizer=l2(l2_reg)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(SpatialTransformer(localization_net=locnet(),
                                 output_size=(32, 32)))
    model.add(Conv2D(16, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Dense(43, activation='softmax'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    return model


def conv_model_no_color_adjust(input_shape=(32, 32, 3)):

    l2_reg = 0.01

    model = Sequential()
    model.add(Lambda(
        lambda x: x * 2 - 1.,
        input_shape=(32, 32, 3),
        output_shape=(32, 32, 3)))
    # model.add(BatchNormalization())
    model.add(SpatialTransformer(localization_net=locnet_v3(),
                                 output_size=(32, 32)))
    model.add(Conv2D(16, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Dense(43, activation='softmax'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    return model


def build_cnn():
    """

    """

    L2_LAMBDA = 1e-4
    l2_reg = keras.regularizers.l2(L2_LAMBDA)

    # Build model
    inpt = keras.layers.Input(shape=(32, 32, 3))
    rescl = Lambda(lambda x: x * 2 - 1., output_shape=(32, 32, 3))(inpt)
    conv1 = keras.layers.Convolution2D(
        16, (5, 5), padding='same', activation='relu')(rescl)
    drop1 = keras.layers.Dropout(rate=0.1)(conv1)
    conv2 = keras.layers.Convolution2D(
        32, (5, 5), padding='same', activation='relu')(drop1)
    drop2 = keras.layers.Dropout(rate=0.2)(conv2)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = keras.layers.Convolution2D(
        64, (5, 5), padding='same', activation='relu')(pool1)
    drop3 = keras.layers.Dropout(rate=0.3)(conv3)
    conv4 = keras.layers.Convolution2D(
        128, (5, 5), padding='same', activation='relu')(drop3)
    drop4 = keras.layers.Dropout(rate=0.3)(conv4)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    flat = keras.layers.Flatten()(pool2)
    dense1 = keras.layers.Dense(200, activation='relu',
                                kernel_regularizer=l2_reg)(flat)
    drop5 = keras.layers.Dropout(rate=0.5)(dense1)
    dense2 = keras.layers.Dense(200, activation='relu',
                                kernel_regularizer=l2_reg)(drop5)
    drop6 = keras.layers.Dropout(rate=0.5)(dense2)
    output = keras.layers.Dense(
        43, activation='softmax', kernel_regularizer=l2_reg)(drop6)
    model = keras.models.Model(inputs=inpt, outputs=output)

    # Specify optimizer
    adam = keras.optimizers.Adam(
        lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_cnn_large():
    """

    """

    L2_LAMBDA = 1e-4
    l2_reg = keras.regularizers.l2(L2_LAMBDA)

    # Build model
    inpt = keras.layers.Input(shape=(32, 32, 3))
    rescl = Lambda(lambda x: x * 2 - 1., output_shape=(32, 32, 3))(inpt)
    conv1 = keras.layers.Convolution2D(
        32, (5, 5), padding='same', activation='relu')(rescl)
    drop1 = keras.layers.Dropout(rate=0.1)(conv1)
    conv2 = keras.layers.Convolution2D(
        32, (5, 5), padding='same', activation='relu')(drop1)
    drop2 = keras.layers.Dropout(rate=0.2)(conv2)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = keras.layers.Convolution2D(
        64, (5, 5), padding='same', activation='relu')(pool1)
    drop3 = keras.layers.Dropout(rate=0.3)(conv3)
    conv4 = keras.layers.Convolution2D(
        64, (5, 5), padding='same', activation='relu')(drop3)
    drop4 = keras.layers.Dropout(rate=0.3)(conv4)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    flat = keras.layers.Flatten()(pool2)
    dense1 = keras.layers.Dense(200, activation='relu',
                                kernel_regularizer=l2_reg)(flat)
    drop5 = keras.layers.Dropout(rate=0.5)(dense1)
    output = keras.layers.Dense(
        43, activation='softmax', kernel_regularizer=l2_reg)(drop5)
    model = keras.models.Model(inputs=inpt, outputs=output)

    # Specify optimizer
    adam = keras.optimizers.Adam(
        lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_cnn_no_stn():
    """

    """

    l2_reg = 0.01

    model = Sequential()
    model.add(Lambda(
        lambda x: x * 2 - 1.,
        input_shape=(32, 32, 3),
        output_shape=(32, 32, 3)))
    model.add(Conv2D(16, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding='same',
                     activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Dense(43, activation='softmax'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    return model


def template_match_nn():

    l2_reg = 0.01

    model = Sequential()
    model.add(Lambda(
        lambda x: x * 2 - 1.,
        input_shape=(32, 32, 3),
        output_shape=(32, 32, 3)))
    model.add(SpatialTransformer(localization_net=locnet(),
                                 output_size=(32, 32)))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(loss='mean_absolute_error', optimizer=adam)

    return model
