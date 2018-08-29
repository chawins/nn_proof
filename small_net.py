from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda,
                          MaxPooling2D)
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from parameters import *
from stn.spatial_transformer import SpatialTransformer
from stn.conv_model import locnet_v3


def create_simple_cnn(pos, stn_weight=None):

    top, bot, left, right = pos
    height = bot - top
    width = right - left

    model = Sequential()
    model.add(Lambda(
        lambda x: x*2 - 1.,
        input_shape=(32, 32, 3),
        output_shape=(32, 32, 3)))
    # Add spartial transformer part
    model.add(SpatialTransformer(localization_net=locnet_v3(),
                                 output_size=(32, 32),
                                 trainable=False,
                                 weights=stn_weight))
    # model.add(Cropping2D(cropping=((top, 32 - bot), (left, 32 - right)),
    #                      input_shape=(32, 32, 3)))
    model.add(Cropping2D(cropping=((top, 32 - bot), (left, 32 - right))))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08, decay=0.01)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    # model.compile(loss=weighted_cross_entropy_loss,
    #               optimizer=adam, metrics=['accuracy'])
    return model


def train_simple_cnn(stn_weight, pos, y, X_train, y_train, X_val, y_val,
                     save_path="./keras_weights/temp.hdf5"):

    # Balance train/val set
    X_train_bal, y_train_bal = gen_balance_data(X_train, y_train, y, r=1)
    X_val_bal, y_val_bal = gen_balance_data(X_val, y_val, y, r=1)

    # ---------------- Train model ----------------- #
    checkpointer = ModelCheckpoint(
        filepath=save_path, verbose=1, save_best_only=True,
        save_weights_only=True)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                              verbose=0, mode='auto', baseline=None)

    model = create_simple_cnn(pos, stn_weight)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    # model.fit(X_train_d, y_train_d,
    #           batch_size=128,
    #           epochs=40,
    #           verbose=1,
    #           shuffle=True,
    #           callbacks=[checkpointer, earlystop],
    #           validation_data=(X_val_d, y_val_d))
    # print("Train: ", model.evaluate(X_train_d, y_train_d))
    # print("Val: ", model.evaluate(X_val_d, y_val_d))

    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        channel_shift_range=0.1)
    # for i in range(10):
    #     x = X_train_d[40000 + i]
    #     plt.imshow(x/2 + 0.5)
    #     plt.show()
    #     x = datagen.random_transform(x)
    #     plt.imshow(x/2 + 0.5)
    #     plt.show()
    batch_size = 128
    model.fit_generator(datagen.flow(X_train_bal, y_train_bal, batch_size=batch_size),
                        steps_per_epoch=len(X_train_bal)/batch_size,
                        epochs=40,
                        verbose=1,
                        shuffle=True,
                        callbacks=[checkpointer, earlystop],
                        validation_data=(X_val_bal, y_val_bal))
    model.load_weights(save_path)
    print("Train: ", model.evaluate(X_train_bal, y_train_bal))
    print("Val: ", model.evaluate(X_val_bal, y_val_bal))

    return model


def eval_simple_cnn(model, stn_fnc, y, X_test, y_test):
    """
    Evaluates small CNN. Returns accuracy, false positive rate, false negative 
    rate. Class '0' is negative (no feature detected), '1' is positive (feature
    detected).
    """

    y_test_d = (y_test == y).astype(int)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    acc = np.sum(y_pred == y_test_d) / len(y_pred)
    fpr = np.sum(np.logical_and(y_pred != y_test_d, y_test_d == 0)
                 ) / np.sum(y_test_d == 0)
    fnr = np.sum(np.logical_and(y_pred != y_test_d, y_test_d == 1)
                 ) / np.sum(y_test_d == 1)

    return acc, fpr, fnr


def weighted_cross_entropy_loss(y_true, y_pred):

    WEIGHT_RATIO = 1.

    class_weights = tf.constant([[1., WEIGHT_RATIO]])
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_true, logits=y_pred)
    loss = weights * loss
    return tf.reduce_mean(loss)


def gen_balance_data(X_train, y_train, y, r=1):

    n_train = len(X_train)
    n_bal = int(n_train*r)
    X_eq = np.zeros((n_train + n_bal, 32, 32, 3))
    X_eq[:n_train] = np.copy(X_train)
    ind = np.where(y_train == y)[0]
    rnd_ind = np.random.choice(ind, size=n_bal)
    X_eq[n_train:] = np.copy(X_train[rnd_ind])

    y_eq = np.zeros(n_train + n_bal)
    y_eq[ind] = 1
    y_eq[n_train:] = 1

    return X_eq, y_eq
