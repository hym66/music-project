import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from utils import *



def abs_backend(inputs):
    return K.abs(inputs)


def expand_dim_backend(inputs):
    return K.expand_dims(K.expand_dims(inputs, 1), 1)


def sign_backend(inputs):
    return K.sign(inputs)


def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels) // 2    # 加了abs
    inputs = K.expand_dims(inputs, -1)
    inputs = K.spatial_3d_padding(inputs, ((0, 0), (0, 0), (pad_dim, pad_dim)), 'channels_last')
    return K.squeeze(inputs, -1)


# Residual Shrinakge Block
def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    for i in range(nb_blocks):

        identity = residual

        if not downsample:
            downsample_strides = 1

        residual = BatchNormalization(trainable=True)(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, strides=(downsample_strides, downsample_strides),
                          padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-3))(residual)

        residual = BatchNormalization(trainable=True)(residual)
        residual = Activation('relu')(residual)

        residual = Conv2D(out_channels, 3, padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-3))(residual)

        # Calculate global means
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling2D()(residual_abs)

        abs_mean = Dropout(0.6)(abs_mean)   # 新加的

        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-3))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu')(scales)

        scales = Dropout(0.55)(scales)

        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-2))(scales)
        scales = Lambda(expand_dim_backend)(scales)

        # Calculate thresholds
        thres = tf.keras.layers.multiply([abs_mean, scales])

        # Soft thresholding
        sub = tf.keras.layers.subtract([residual_abs, thres])
        zeros = tf.keras.layers.subtract([sub, sub])
        n_sub = tf.keras.layers.maximum([sub, zeros])
        residual = tf.keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])

        # Downsampling using the pooL-size of (1, 1)
        if downsample_strides > 1:
            identity = AveragePooling2D(pool_size=(1, 1), strides=(2, 2))(identity)

        # Zero_padding to match channels
        if in_channels != out_channels:
            identity = Lambda(pad_backend, arguments={'in_channels': in_channels, 'out_channels': out_channels})(
                identity)
        residual = tf.keras.layers.add([residual, identity])

    return residual



def DRSN():
    input_shape = (eachInput_rows, eachInput_cols, 1)
    inputs = Input(input_shape)
    net = Conv2D(8, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(inputs)

    # 卷积核大小都为3*3
    # downsample为True代表卷积步长为2
    #           为False代表卷积步长为1
    #           默认为False即卷积步长为1
    #
    # 第三个参数为卷积核个数，即out_channel的个数

    net = residual_shrinkage_block(net, 1, 64)
    net = residual_shrinkage_block(net, 1, 64)
    net = residual_shrinkage_block(net, 1, 128, downsample=True)
    net = residual_shrinkage_block(net, 1, 128)
    net = residual_shrinkage_block(net, 1, 256, downsample=True)
    net = residual_shrinkage_block(net, 1, 256)
    net = residual_shrinkage_block(net, 1, 512, downsample=True)
    net = residual_shrinkage_block(net, 1, 512)

    # net = residual_shrinkage_block(net, 1, 128)
    # net = residual_shrinkage_block(net, 1, 128, downsample=True)
    # net = residual_shrinkage_block(net, 1, 256)
    # net = residual_shrinkage_block(net, 1, 256, downsample=True)
    # net = residual_shrinkage_block(net, 1, 512)
    # net = residual_shrinkage_block(net, 1, 512, downsample=True)



    net = BatchNormalization(trainable=True)(net)
    net = Activation('relu')(net)
    net = GlobalAveragePooling2D()(net)
    outputs = Dense(CLASS_NUM, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(net)
    model = Model(inputs=inputs, outputs=outputs)

    return model
