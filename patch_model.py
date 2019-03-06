"""
This model makes classification based on random patches of an image.
We try to detect/defense against adversarial examples by requiring that
the prediceted class from each random set of patches must "agree."

Some code on this file in taken from
https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
"""

import keras
import tensorflow as tf
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, ReduceLROnPlateau)
from keras.datasets import cifar10, mnist
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Concatenate, Conv2D, Cropping2D, Dense, Dropout,
                          Flatten, Input, Lambda, Layer, MaxPooling2D, Reshape,
                          ZeroPadding2D)
from keras.models import Model, Sequential, load_model
from keras.regularizers import l2

from stn.conv_model import locnet_v3
from stn.spatial_transformer import SpatialTransformer

HEIGHT = 32
WIDTH = 32
CHANNEL = 3
NUM_CLASSES = 43


class MyLogger(keras.callbacks.Callback):
    """Callback to log training info using a logger"""

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_train_begin(self, logs={}):
        self.logger.info(str(self.params))
        self.logger.info(' epoch | loss  , acc    | val_loss, val_acc')
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.logger.info(' {:5d} | {:.4f}, {:.4f} | {:8.4f}, {:7.4f}'.format(
            epoch, logs['loss'], logs['acc'], logs['val_loss'], logs['val_acc']))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class RandomCropLayer(Layer):
    """A custom layer to randomly 2D-crop a patch from input"""

    def __init__(self, size, **kwargs):
        self.size = size
        pad_size = size // 2
        self.paddings = tf.constant(
            [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        super(RandomCropLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RandomCropLayer, self).build(input_shape)

    def call(self, x):
        padded = tf.pad(x, self.paddings)
        cropped = tf.random_crop(
            padded, [tf.shape(x)[0], self.size, self.size, CHANNEL])
        return cropped

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.size, self.size, input_shape[-1])


def lr_schedule_200(epoch):
    """
    Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def lr_schedule_20(epoch):
    """
    Learning Rate Schedule for 20 epochs
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 18:
        lr *= 0.5e-3
    elif epoch > 16:
        lr *= 1e-3
    elif epoch > 12:
        lr *= 1e-2
    elif epoch > 8:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 l2_reg=1e-4):
    """
    2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(l2_reg))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def apply_layers(input, layers_list):

    y = input
    for layer in layers_list:
        y = layer(y)
    return y


def build_resnet_layer(num_filters=16,
                       kernel_size=3,
                       strides=1,
                       activation='relu',
                       batch_normalization=True,
                       conv_first=True,
                       l2_reg=1e-4):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(l2_reg))
    # TODO: there's tensorflow bug when batchnorm is reused
    # bn = BatchNormalization()
    act = Activation(activation)

    layers_list = []
    # if conv_first:
    #     layers_list.append(conv)
    #     if batch_normalization:
    #         layers_list.append(bn)
    #     if activation is not None:
    #         layers_list.append(act)
    # else:
    #     if batch_normalization:
    #         layers_list.append(bn)
    #     if activation is not None:
    #         layers_list.append(act)
    #     layers_list.append(conv)
    layers_list = [conv]
    if activation is not None:
        layers_list.append(act)

    return layers_list


def build_resnet_v2(depth, l2_reg=1e-4):

    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    first_layers = build_resnet_layer(num_filters=num_filters_in,
                                      conv_first=True,
                                      l2_reg=l2_reg)
    all_layers_list = [first_layers]

    # Instantiate the stack of residual units
    for stage in range(3):
        stage_list = []
        for res_block in range(num_res_blocks):
            layers_list = []
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2     # downsample

            # bottleneck residual unit
            layers = build_resnet_layer(num_filters=num_filters_in,
                                        kernel_size=1,
                                        strides=strides,
                                        activation=activation,
                                        batch_normalization=batch_normalization,
                                        conv_first=False,
                                        l2_reg=l2_reg)
            layers_list.append(layers)
            layers = build_resnet_layer(num_filters=num_filters_in,
                                        conv_first=False,
                                        l2_reg=l2_reg)
            layers_list.append(layers)
            layers = build_resnet_layer(num_filters=num_filters_out,
                                        kernel_size=1,
                                        conv_first=False,
                                        l2_reg=l2_reg)
            layers_list.append(layers)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                layers = build_resnet_layer(num_filters=num_filters_out,
                                            kernel_size=1,
                                            strides=strides,
                                            activation=None,
                                            batch_normalization=False,
                                            l2_reg=l2_reg)
                layers_list.append(layers)

            stage_list.append(layers_list)

        all_layers_list.append(stage_list)
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    dense = Dense(NUM_CLASSES,
                  activation=None,
                  kernel_initializer='he_normal')
    # final_layers = [BatchNormalization(), Activation('relu'), Flatten(), dense]
    all_layers_list.append([Flatten(), dense])
    # all_layers_list.append([MaxPooling2D((2, 2)), Flatten(), dense])

    return all_layers_list


def apply_resnet(input, all_layers_list):

    # Apply first layers
    x = apply_layers(input, all_layers_list[0])

    # Apply layers from stage 1 and 2
    for stage in all_layers_list[1:-1]:
        for i, block in enumerate(stage):
            y = apply_layers(x, block[0])
            for layer in block[1:-1]:
                y = apply_layers(y, layer)
            if i == 0:
                x = apply_layers(x, block[-1])
            else:
                y = apply_layers(y, block[-1])
            x = keras.layers.Add()([x, y])

    # Apply final layers
    y = apply_layers(x, all_layers_list[-1])

    return y


def build_patch_model_resnet(patch_size=8,
                             use_stn=True,
                             stn_weight=None,
                             l2_reg=1e-4,
                             patch_scheme='random',
                             num_patches_total=16,
                             use_batchnorm=False):
    """
    Build PatchNet with ResNet blocks.

    # Arguments
        patch_size (int): height and width of the patch
        use_stn (bool): whether to use STN before PatchNet. If True, the
            pretrained weights must be provided as stn_weight
        stn_weight (np.array): STN weights, required if use_stn is True
        l2_reg (float): l2 weight regularization constant
        patch_scheme (str): must be one of the followings
            'no-overlap' (image is splitted into a non-overlapping grid;
                          'valid' padding),
            'random' (patches are chosen randomly; 'same' padding),
            'all' (all pixels are used as center of a patch; 'same' padding)
        num_patches_total (int): the number of total patches to use, required
            if patch_scheme is 'random'

    # Returns
        model (keras model): PatchNet as uncompiled keras model
    """

    x = Input(shape=[HEIGHT, WIDTH, CHANNEL])
    # scale to [-1, 1]
    v = Lambda(lambda x: x * 2 - 1., output_shape=(HEIGHT, WIDTH, CHANNEL))(x)
    if use_stn:
        v = SpatialTransformer(localization_net=locnet_v3(),
                               output_size=(HEIGHT, WIDTH),
                               trainable=False,
                               weights=stn_weight)(v)

    if patch_scheme == 'no-overlap':
        num_patches = HEIGHT // patch_size
        num_patches_total = num_patches**2
    elif patch_scheme == 'random':
        random_crop_layer = [RandomCropLayer(patch_size)]
    elif patch_scheme == 'all':
        num_patches_total = HEIGHT * WIDTH
        v = ZeroPadding2D(padding=(patch_size // 2, patch_size // 2))(v)
    else:
        raise ValueError("patch_scheme must be one of the followings:" +
                         "'no-overlap', 'random', 'all'")

    # Create the patch network
    layers_list = build_resnet_v2(20, l2_reg=l2_reg)

    output = []
    for i in range(num_patches_total):
        if patch_scheme == 'no-overlap':
            h = i // num_patches
            w = i % num_patches
            top_crop = h * patch_size
            bottom_crop = HEIGHT - top_crop - patch_size
            left_crop = w * patch_size
            right_crop = WIDTH - left_crop - patch_size
            u = Cropping2D(
                ((top_crop, bottom_crop), (left_crop, right_crop)))(v)
        elif patch_scheme == 'random':
            u = apply_layers(v, random_crop_layer)
        elif patch_scheme == 'all':
            top_crop = i // HEIGHT
            left_crop = i % WIDTH
            bottom_crop = HEIGHT - top_crop - (patch_size % 2)
            right_crop = WIDTH - left_crop - (patch_size % 2)
            u = Cropping2D(
                ((top_crop, bottom_crop), (left_crop, right_crop)))(v)
        # Separate batch norm for each patch
        if use_batchnorm:
            u = BatchNormalization()(u)
        u = apply_resnet(u, layers_list)
        output.append(u)

    merge = Concatenate()(output)
    reshape = Reshape([num_patches_total, NUM_CLASSES])(merge)
    mean = Lambda(lambda x: tf.reduce_mean(x, 1),
                  output_shape=(NUM_CLASSES, ))(reshape)
    model = keras.models.Model(inputs=x, outputs=mean)
    model_map = keras.models.Model(inputs=x, outputs=reshape)

    return model, model_map


def build_patch_model(patch_size=8,
                      use_stn=True,
                      stn_weight=None):

    num_patches = HEIGHT // patch_size
    x = Input(shape=[HEIGHT, WIDTH, CHANNEL])
    # scale to [-1, 1]
    v = Lambda(lambda x: x * 2 - 1., output_shape=(HEIGHT, WIDTH, CHANNEL))(x)
    if use_stn:
        v = SpatialTransformer(localization_net=locnet_v3(),
                               output_size=(HEIGHT, WIDTH),
                               trainable=False,
                               weights=stn_weight)(v)

    # Create the patch network
    conv1 = Conv2D(32, (5, 5), padding='same', activation="relu")
    conv2 = Conv2D(64, (3, 3), padding='same', activation="relu")
    conv3 = Conv2D(128, (3, 3), padding='same', activation="relu")
    # bn = BatchNormalization()
    flat = Flatten()
    dense1 = Dense(256, activation="relu")
    dense2 = Dense(256, activation="relu")
    dense3 = Dense(NUM_CLASSES, activation=None)

    output = []
    for i in range(num_patches**2):
        h = i // num_patches
        w = i % num_patches
        top_crop = h * patch_size
        bottom_crop = HEIGHT - top_crop - patch_size
        left_crop = w * patch_size
        right_crop = WIDTH - left_crop - patch_size
        u = Cropping2D(((top_crop, bottom_crop), (left_crop, right_crop)))(v)
        # u = BatchNormalization()(u)
        u = conv1(u)
        u = conv2(u)
        u = conv3(u)
        u = flat(u)
        # u = bn(u)
        u = dense1(u)
        u = dense2(u)
        u = dense3(u)
        output.append(u)

    merge = Concatenate()(output)
    reshape = Reshape([num_patches**2, NUM_CLASSES])(merge)
    mean = Lambda(lambda x: tf.reduce_mean(x, 1),
                  output_shape=(NUM_CLASSES, ))(reshape)
    model = keras.models.Model(inputs=x, outputs=mean)

    return model


def get_heatmap(model, images, label=None):
    """
    Get heatmap of model's output (before averaging). If label is specified,
    only return heatmap for that label. Otherwise, heatmap for all labels is
    returned.
    """

    output = model.predict(images, verbose=0)
    if label is not None:
        output = output[:, :, label]
        return output.reshape(images[:-1])
    else:
        return output.reshape(images[:-1] + (NUM_CLASSES, ))


# =============================== Legacy Code =============================== #


def resnet_v1(input_shape, depth, num_classes=10, l2_reg=1e-4):
    """
    ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, l2_reg=l2_reg)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             l2_reg=l2_reg)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             l2_reg=l2_reg)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 l2_reg=l2_reg)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    y = Dense(num_classes,
              activation=None,
              kernel_initializer='he_normal')(y)
    outputs = Activation('softmax')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10, l2_reg=1e-4):
    """
    ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True,
                     l2_reg=l2_reg)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2     # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False,
                             l2_reg=l2_reg)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False,
                             l2_reg=l2_reg)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False,
                             l2_reg=l2_reg)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 l2_reg=l2_reg)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
