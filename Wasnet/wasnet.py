# -*- coding: utf-8 -*-
"""
 @Time    : 19-11-21 下午4:30
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : wasnet.py
"""
from keras import backend as K
from keras import layers, backend ,models
from keras.utils import plot_model
from attention_module import attach_attention_module

if backend.image_data_format( ) == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1


def relu6(x):
    return K.relu(x, max_value = 6.0)


def hard_swish(x):
    return x * K.relu(x + 3.0, max_value = 6.0) / 6.0


def return_activation(x, nl):
    if nl == 'HS':
        x = layers.Activation(hard_swish)(x)
    if nl == 'RE':
        x = layers.Activation(relu6)(x)
    return x


def WasterNet(input_shape = None, classes = 40):
    input = layers.Input(shape = input_shape)
    x = layers.ZeroPadding2D(padding = (3, 3))(input)

    x = layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1))(x)
    x = layers.BatchNormalization(axis = bn_axis)(x)
    x = return_activation(x, 'HS')
    x = layers.ZeroPadding2D(padding = (1, 1))(x)

    x = layers.MaxPooling2D((3, 3), strides = (2, 2))(x)

    x = _wasnet_block(x, filter = 64, strides = (2, 2), nl = 'RE')

    x = _wasnet_block(x, filter = 128, strides = (2, 2), nl = 'HS')

    x = _wasnet_block(x, filter = 256, strides = (2, 2), nl = 'HS')

    x = _wasnet_block(x, filter = 512, strides = (2, 2), nl = 'HS')


    x = layers.SeparableConv2D(512, (3, 3), padding = 'same')(x)
    x = layers.BatchNormalization(axis = bn_axis)(x)
    x = return_activation(x, 'HS')

    x = layers.GlobalAveragePooling2D( )(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(classes, activation = 'softmax')(x)
    model = models.Model(input, x, )

    # model.summary( )
    return model


def _wasnet_block(x, filter, strides = (2, 2), nl = 'RE'):

    residual = layers.Conv2D(filter, kernel_size = (1, 1), strides = strides, padding = 'same')(x)
    residual = layers.BatchNormalization(axis = bn_axis)(residual)

    cbam = attach_attention_module(residual, attention_module = 'cbam_block')

    x = layers.SeparableConv2D(filter, (3, 3), padding = 'same')(x)
    x = layers.BatchNormalization(axis = bn_axis)(x)
    x = return_activation(x, nl)
    x = layers.SeparableConv2D(filter, (3, 3), padding = 'same')(x)
    x = layers.BatchNormalization(axis = bn_axis)(x)

    x = layers.MaxPooling2D((3, 3), strides = strides, padding = 'same')(x)
    x = layers.add([x, residual, cbam])

    return x




if __name__ == '__main__':
    input_shape = (224, 224, 3)
    classes = 40
    model = WasterNet(input_shape = input_shape, classes = 10000)
    plot_model(model, to_file = 'wasnet.png', show_shapes = True, show_layer_names = True)
