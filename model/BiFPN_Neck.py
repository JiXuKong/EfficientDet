import numpy as np
import tensorflow as tf
import tensorflow.layers as layers
from model.EfficientNet.efficientnet import conv_kernel_initializer

MOMENTUM = 0.997
EPSILON = 1e-4


def wBiFPNAdd(inputs, name, epsilon=1e-4, **kwargs):
    w = tf.get_variable(name=name,
                                 shape=(num_in,),
                                 initializer=tf.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)
    w = tf.nn.relu(w)
    x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
    x = x / (tf.reduce_sum(w) + epsilon)
    return x

def UpSampling2D(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs

def SeparableConvBlock(x, num_channels, kernel_size, strides, is_training, freeze_bn=False, name):
    x = layers.separable_conv2d(x, num_channels,
                            kernel_size,
                            strides,
                            padding='same',
                            use_bias=True,
                            trainable = is_training,
                            depthwise_initializer=conv_kernel_initializer,
                            name=f'{name}/conv')
    x = layers.BatchNormalization(x, momentum=MOMENTUM, epsilon=EPSILON, trainable=freeze_bn, name=f'{name}/bn')
    return x

def ConvBlock(x, num_channels, kernel_size, strides, is_training, freeze_bn=False, name):
    x = layers.conv2d(x, num_channels, 
                      kernel_size,
                      strides=strides,
                      padding='same',
                      use_bias=True,
                      trainable = is_training,
                      kernel_initializer=conv_kernel_initializer,
                      name='{}_conv'.format(name))
    x = layers.BatchNormalization(x, momentum=MOMENTUM, epsilon=EPSILON, trainable=freeze_bn, name=f'{name}/bn')
    return tf.nn.relu(x, name='{}_relu'.format(name))

def build_wBiFPN(features, num_channels, id, use_wAdd=False, is_training=True, freeze_bn=True):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5
        P6_in = layers.conv2d(C5, num_channels, kernel_size=1, padding='same', trainable = is_training, name='resample_p6/conv2d')
        P6_in = layers.BatchNormalization(P6_in, momentum=MOMENTUM, epsilon=EPSILON, trainable=freeze_bn, name='resample_p6/bn')
        P6_in = layers.MaxPooling2D(P6_in, pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')
        P7_in = layers.MaxPooling2D(P6_in, pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')
        P7_U = UpSampling2D(P7_in, tf.shape(P6_in)[1:3])
        if use_wAdd:
            P6_td = wBiFPNAdd([P6_in, P7_U], name=f'fpn_cells/cell_{id}/fnode0/add')
        else:
            P6_td = tf.add(P6_in, P7_U, name=f'fpn_cells/cell_{id}/fnode0/add')
        P6_td = tf.nn.swish(P6_td)
        P6_td = SeparableConvBlock(P6_td, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')
        P5_in_1 = layers.conv2d(P5_in, num_channels, kernel_size=1, padding='same', trainable = is_training,
                                name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')
        P5_in_1 = layers.BatchNormalization(P5_in_1, momentum=MOMENTUM, epsilon=EPSILON, freeze_bn = freeze_bn,
                                            name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')
        P6_U = UpSampling2D(P6_td, tf.shape(P5_in_1)[1:3])
        if use_wAdd:
            P5_td = wBiFPNAdd([P5_in_1, P6_U], name=f'fpn_cells/cell_{id}/fnode1/add')
        else:
            P5_td = tf.add(P5_in_1, P6_U, name=f'fpn_cells/cell_{id}/fnode1/add')
        P5_td = tf.nn.swish(P5_td)
        P5_td = SeparableConvBlock(P5_td, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')
        P4_in_1 = layers.conv2d(P4_in, num_channels, kernel_size=1, padding='same', trainable = is_training,
                                name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')()
        P4_in_1 = layers.BatchNormalization(P4_in_1, momentum=MOMENTUM, epsilon=EPSILON, trainable=freeze_bn, 
                                            name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')
        P5_U = layers.UpSampling2D(P5_td, tf.shape(P4_in_1)[1:3])
        if use_wAdd:
            P4_td = wBiFPNAdd([P4_in_1, P5_U], name=f'fpn_cells/cell_{id}/fnode2/add')
        else:
            P4_td = tf.add(P4_in_1, P5_U, name=f'fpn_cells/cell_{id}/fnode2/add')
        P4_td = tf.nn.swish(P4_td)
        P4_td = SeparableConvBlock(P4_td, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')
        P3_in = layers.conv2d(P3_in, num_channels, kernel_size=1, padding='same', trainable = is_training,
                              name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')
        P3_in = layers.BatchNormalization(P3_in, momentum=MOMENTUM, epsilon=EPSILON, trainable=freeze_bn,
                                          name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')
        
        P4_U = layers.UpSampling2D(P4_td, tf.shape(P3_in)[1:3])
        if use_wAdd:
            P3_out = wBiFPNAdd([P3_in, P4_U], name=f'fpn_cells/cell_{id}/fnode3/add')
        else:
            P3_out = tf.add(P3_in, P4_U, name=f'fpn_cells/cell_{id}/fnode3/add')
        P3_out = tf.nn.swish(P3_out)
        P3_out = SeparableConvBlock(P3_out, num_channels=num_channels, kernel_size=3, strides=1,is_training=is_training, freeze_bn=freeze_bn,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')()
        P4_in_2 = layers.conv2d(num_channels, kernel_size=1, padding='same', trainable=is_training,
                                name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = layers.BatchNormalization(P4_in_2, momentum=MOMENTUM, epsilon=EPSILON, trainable=freeze_bn,
                                            name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')
        P3_D = layers.MaxPooling2D(P3_out, pool_size=3, strides=2, padding='same')
        if use_wAdd:
            P4_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in_2, P4_td, P3_D])
        else:
            P4_out = tf.add_n([P4_in_2, P4_td, P3_D], name=f'fpn_cells/cell_{id}/fnode4/add')
        P4_out = tf.nn.swish(P4_out)
        P4_out = SeparableConvBlock(P4_out, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')()

        P5_in_2 = layers.conv2d(P5_in, num_channels, kernel_size=1, padding='same', trainable=is_training,
                                name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')
        P5_in_2 = layers.BatchNormalization(P5_in_2, momentum=MOMENTUM, epsilon=EPSILON, trainable=freeze_bn,
                                            name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')
        P4_D = layers.MaxPooling2D(P4_out, pool_size=3, strides=2, padding='same')
        if use_wAdd:
            P5_out = wBiFPNAdd([P5_in_2, P5_td, P4_D], name=f'fpn_cells/cell_{id}/fnode5/add')()
        else:
            P5_out = tf.add_n([P5_in_2, P5_td, P4_D], name=f'fpn_cells/cell_{id}/fnode5/add')
        P5_out = tf.nn.swish(P5_out)
        P5_out = SeparableConvBlock(P5_out, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')

        P5_D = layers.MaxPooling2D(P5_out, pool_size=3, strides=2, padding='same')
        if use_wAdd:
            P6_out = wBiFPNAdd([P6_in, P6_td, P5_D], name=f'fpn_cells/cell_{id}/fnode6/add')
        else:
            P6_out = tf.add([P6_in, P6_td, P5_D], name=f'fpn_cells/cell_{id}/fnode6/add')
        P6_out = tf.nn.swish(P6_out)
        P6_out = SeparableConvBlock(P6_out, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')

        P6_D = layers.MaxPooling2D(P6_out, pool_size=3, strides=2, padding='same')
        if use_wAdd:
            P7_out = wBiFPNAdd([P7_in, P6_D], name=f'fpn_cells/cell_{id}/fnode7/add')
        else:
            P7_out = tf.add([P7_in, P6_D], name=f'fpn_cells/cell_{id}/fnode7/add')
        P7_out = tf.nn.swish(P7_out)
        P7_out = SeparableConvBlock(P7_out, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')

    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P7_U = UpSampling2D(P7_in, tf.shape(P6_in)[1:3])
        if use_wAdd:
            P6_td = wBiFPNAdd([P6_in, P7_U], name=f'fpn_cells/cell_{id}/fnode0/add')
        else:
            P6_td = tf.add([P6_in, P7_U], name=f'fpn_cells/cell_{id}/fnode0/add')
        P6_td = tf.nn.swish(P6_td)
        P6_td = SeparableConvBlock(P6_td， num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')
        P6_U = UpSampling2D(P6_td， tf.shape(P5_in)[1:3])
        if use_wAdd:
            P5_td = wBiFPNAdd([P5_in, P6_U], name=f'fpn_cells/cell_{id}/fnode1/add')
        else:
            P5_td = wBiFPNAdd([P5_in, P6_U], name=f'fpn_cells/cell_{id}/fnode1/add')
        P5_td = tf.nn.swish(P5_td)
        P5_td = SeparableConvBlock(P5_td, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')
        P5_U = UpSampling2D(P5_td, tf.shape(P4_in)[1:3])
        if use_wAdd:
            P4_td = wBiFPNAdd([P4_in, P5_U], name=f'fpn_cells/cell_{id}/fnode2/add')
        else:
            P4_td = tf.add([P4_in, P5_U], name=f'fpn_cells/cell_{id}/fnode2/add')
        P4_td = tf.nn.swish(P4_td)
        P4_td = SeparableConvBlock(P4_td, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')
        P4_U = layers.UpSampling2D(P4_td, tf.shape(P3_in)[1:3])
        if use_wAdd:
            P3_out = wBiFPNAdd([P3_in, P4_U], name=f'fpn_cells/cell_{id}/fnode3/add')
        else:
            P3_out = tf.add([P3_in, P4_U], name=f'fpn_cells/cell_{id}/fnode3/add')
        P3_out = tf.nn.swish(P3_out)
        P3_out = SeparableConvBlock(P3_out, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')
        P3_D = layers.MaxPooling2D(P3_out, pool_size=3, strides=2, padding='same')
        if use_wAdd:
            P4_out = wBiFPNAdd([P4_in, P4_td, P3_D], name=f'fpn_cells/cell_{id}/fnode4/add')
        else:
            P4_out = tf.add_n([P4_in, P4_td, P3_D], name=f'fpn_cells/cell_{id}/fnode4/add')
        P4_out = tf.nn.swish(P4_out)
        P4_out = SeparableConvBlock(P4_out, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')

        P4_D = layers.MaxPooling2D(P4_out, pool_size=3, strides=2, padding='same')
        if use_wAdd:
            P5_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
        else:
            P5_out = tf.add_n([P5_in, P5_td, P4_D], name=f'fpn_cells/cell_{id}/fnode5/add')
        P5_out = tf.nn.swish(P5_out)
        P5_out = SeparableConvBlock(P5_out, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')

        P5_D = layers.MaxPooling2D(P5_out, pool_size=3, strides=2, padding='same')
        if use_wAdd:
            P6_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        else:
            P6_out = tf.add([P6_in, P6_td, P5_D], name=f'fpn_cells/cell_{id}/fnode6/add')
        P6_out = tf.nn.swish(P6_out)
        P6_out = SeparableConvBlock(P6_out, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')

        P6_D = layers.MaxPooling2D(P6_out, pool_size=3, strides=2, padding='same')
        if use_wAdd:
            P7_out = wBiFPNAdd([P7_in, P6_D], name=f'fpn_cells/cell_{id}/fnode7/add')
        else:
            P7_out = tf.add([P7_in, P6_D], name=f'fpn_cells/cell_{id}/fnode7/add')
        P7_out = tf.nn.swish(P7_out)
        P7_out = SeparableConvBlock(P7_out, num_channels=num_channels, kernel_size=3, strides=1, is_training=is_training, freeze_bn=freeze_bn,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')()
    return P3_out, P4_td, P5_td, P6_td, P7_out


