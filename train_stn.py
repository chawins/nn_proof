import os
import pickle

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from lib.utils import load_gtsrb
from parameters import *
from stn.conv_model import conv_model
from stn.conv_model import conv_model_no_color_adjust

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

X_train, y_train, X_val, y_val, X_test, y_test = load_gtsrb()

print("Number of training examples =", X_train.shape[0])
print("Number of validating examples =", X_val.shape[0])
print("Number of testing examples =", X_test.shape[0])
print("Image data shape =", X_train[0].shape)
print("Number of classes =", NUM_LABELS)

batch_size = 128
epochs = 150
# model = conv_model()
model = conv_model_no_color_adjust()
save_path = "./keras_weights/stn_v8.hdf5"

checkpointer = ModelCheckpoint(
    filepath=save_path, verbose=1, save_best_only=True,
    save_weights_only=True)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                          verbose=0, mode='auto', baseline=None)

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.1)
try:
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train)/batch_size, epochs=epochs,
                        verbose=1, shuffle=True,
                        callbacks=[checkpointer, earlystop],
                        validation_data=(X_val, y_val))
# try:
#     model.fit(X_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               validation_data=(X_val, y_val),
#               shuffle=True,
#               callbacks=[checkpointer, earlystop])
except KeyboardInterrupt:
    print("training interrupted")

model.load_weights(save_path)
acc = model.evaluate(X_test, y_test)[1]
print("Test accuracy = {}".format(acc))
