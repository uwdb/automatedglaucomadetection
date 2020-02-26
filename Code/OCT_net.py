import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D,
                                     GlobalAveragePooling2D,
                                     BatchNormalization,
                                     Flatten, Dropout, Dense, ReLU,
                                     AveragePooling2D, Concatenate, Lambda)


# densenet block implementation from https://github.com/taki0112/Densenet-Tensorflow
filters = 12
dropout_rate = 0.2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4
nclasses = 2


def conv_layer(x, filter, kernel, strides=1, name="conv"):
    x = Conv2D(
        filters=filter, kernel_size=kernel, strides=strides,
        padding='same',use_bias=False,name=name+'_conv',
        kernel_initializer=tf.initializers.variance_scaling(
            scale=2.0,
            mode='fan_in',
            distribution="truncated_normal"))(x)
    return x


def bottleneck_layer( x, filters, name):
    x = BatchNormalization(name=name+'_bn_1',axis=-1)(x)
    x = ReLU()(x)
    x = conv_layer(x, filter=4 * filters, kernel=1, name=name+'_conv1')
    x = Dropout(dropout_rate, name=name+'_dropout_first')(x)
    x = BatchNormalization(name=name+"_bn_2")(x)
    x = ReLU()(x)
    x = conv_layer(x, filter=filters, kernel=3, name=name+'_conv2')
    x = Dropout(dropout_rate, name=name+'_dropout_second')(x)
    return x


def transition_layer(x,name):
    x = BatchNormalization(axis=-1)(x)
    x = ReLU()(x)
    in_channel = x.shape[-1]
    in_channel = int(int(in_channel)*0.5)
    x = conv_layer(x, filter=in_channel , kernel=1, name=name+'_conv')
    x = Dropout(0.2)(x)
    x = AveragePooling2D( pool_size=2, strides=2,padding='valid')(x)
    return x


def db(layers_concat):
    x = Concatenate(axis=3)(layers_concat[:])
    x = BatchNormalization(axis=-1)(x)
    x = ReLU()(x)
    x = conv_layer(x, filter=4 * filters, kernel=1)
    x = Dropout(0.2)(x)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU()(x)
    x = conv_layer(x, filter=filters, kernel=3)
    x = Dropout(0.2)(x)
    layers_concat.append(x)
    return x, layers_concat


def dense_block(input_x, nb_layers,name):
    layers_concat = []
    layers_concat.append(input_x)
    x = bottleneck_layer(input_x,filters=12, name=name + '_bottleN_' + str(0))
    layers_concat.append(x)

    for i in range(nb_layers - 1):
        x = Concatenate(axis=-1)(layers_concat[:])
        x = bottleneck_layer(x, filters=12,name=name + '_bottleN_' + str(i + 1))
        layers_concat.append(x)

    x = Concatenate()(layers_concat[:])
    return x


def dense_net_model(nclasses=2):
    inputs = tf.keras.Input(shape=(224, 224, 1))
    x = conv_layer(inputs, filter=2 *12, kernel=7, strides=2,name='conv_1')
    x = MaxPool2D(pool_size=3, strides=2, padding='valid')(x)

    x = dense_block(input_x=x, nb_layers=6,name='dense0')
    x = transition_layer(x,name='trans1')
    x = dense_block(input_x=x, nb_layers=12,name='dense1')
    x = transition_layer(x,name='trans2')
    x = dense_block(input_x=x, nb_layers=48,name='dense3')
    x = transition_layer(x,name='trans3')
    x = dense_block(input_x=x, nb_layers=32,name='dense_final')

    # 100 Layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Flatten()(x)
    output = Dense(
        nclasses,
        kernel_initializer=tf.initializers.variance_scaling(
            scale=2.0,
            mode='fan_in',
            distribution="truncated_normal"))(x)

    return tf.keras.Model(inputs=inputs, outputs=output, name='oct_dense_net')
