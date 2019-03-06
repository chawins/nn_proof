"""Train scrip for PatchNet"""

import gc
import logging
import os
import pickle
import random as rn

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, ReduceLROnPlateau)
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from lib.utils import load_gtsrb
from patch_model import *
from stn.conv_model import conv_model_no_color_adjust


def main():

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    # Set all random seeds
    SEED = 2019
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    rn.seed(SEED)

    # Set experiment id
    exp_id = 13

    # Training parameters
    batch_size = 32
    epochs = 200
    data_augmentation = True
    l1_reg = 0
    l2_reg = 1e-4

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = False

    # Get TF logger
    log_file = 'train_patch_net_exp{}.log'.format(exp_id)
    log = logging.getLogger('train_resnet')
    log.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s %(name)s] %(message)s')
    # Create file handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    log.info(log_file)
    log.info(('PatchNet GTSRB | seed: {}, init_learning_rate: {}, ' +
              'batch_size: {}, l2_reg: {}, l1_reg: {}, epochs: {}, ' +
              'data_augmentation: {}, subtract_pixel_mean: {}').format(
                  SEED, 1e-3, batch_size, l2_reg, l1_reg, epochs,
                  data_augmentation, subtract_pixel_mean))

    # Load GTSRB
    x_train, y_train, x_val, y_val, x_test, y_test = load_gtsrb()

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_val -= x_train_mean
        x_test -= x_train_mean

    log.info('x_train shape: {}'.format(x_train.shape))
    log.info('{} train samples'.format(x_train.shape[0]))
    log.info('{} test samples'.format(x_test.shape[0]))
    log.info('y_train shape: {}'.format(y_train.shape))

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    model = conv_model_no_color_adjust()
    model.load_weights("./keras_weights/stn_v5.hdf5")
    stn_weight = model.layers[1].get_weights()
    # Delete STN to reclaim unused GPU memory
    del model
    gc.collect()

    # model = build_patch_model(patch_size=8,
    #                           use_stn=True,
    #                           stn_weight=stn_weight)
    model, model_map = build_patch_model_resnet(patch_size=8,
                                                use_stn=True,
                                                stn_weight=stn_weight,
                                                l2_reg=l2_reg,
                                                patch_scheme='no-overlap',
                                                num_patches_total=32,
                                                use_batchnorm=False)

    def loss_func(y_true, y_pred):
        softmax = tf.nn.softmax(y_pred, 1)
        loss = -tf.log(tf.reduce_sum(softmax * y_true, 1))
        loss = tf.reduce_mean(loss)
        return loss

    model.compile(loss=loss_func,
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    model.summary(print_fn=lambda x: log.info(x))

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'train_patch_net_exp%d.h5' % exp_id
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True)
    # lr_scheduler = LearningRateScheduler(lr_schedule_20)
    lr_scheduler = LearningRateScheduler(lr_schedule_200)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [checkpoint, lr_reducer, lr_scheduler, MyLogger(log)]

    # Run training, with or without data augmentation.
    if not data_augmentation:
        log.info('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_val, y_val),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        log.info('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=5,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.1,
            # set range for random zoom
            zoom_range=0.1,
            # set range for random channel shifts
            channel_shift_range=0.1,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=False,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_val, y_val),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

    # Score trained model.
    model.load_weights(filepath)
    scores = model.evaluate(x_train, y_train, verbose=1)
    log.info('Train loss: {:.4f}'.format(scores[0]))
    log.info('Train accuracy: {:.4f}'.format(scores[1]))
    scores = model.evaluate(x_val, y_val, verbose=1)
    log.info('Val loss: {:.4f}'.format(scores[0]))
    log.info('Val accuracy: {:.4f}'.format(scores[1]))
    scores = model.evaluate(x_test, y_test, verbose=1)
    log.info('Test loss: {:.4f}'.format(scores[0]))
    log.info('Test accuracy: {:.4f}'.format(scores[1]))


if __name__ == "__main__":
    main()
