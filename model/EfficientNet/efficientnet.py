'''
codes referenced from: https://github.com/qubvel/efficientnet
'''

import numpy as np
import tensorflow as tf
import tensorflow.layers as layers
import collections

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]






def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))

def conv_kernel_initializer(shape, dtype=None):
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def drop_connect(inputs, is_training, drop_connect_rate):
    """Apply drop connect.
    Args:
    inputs: `Tensor` input tensor.
    is_training: `bool` if True, the model is in training mode.
    drop_connect_rate: `float` drop connect rate.
    Returns:
    A output tensor, which should have the same shape as input.
    """
    if not is_training or drop_connect_rate is None or drop_connect_rate == 0:
        return inputs

    keep_prob = 1.0 - drop_connect_rate
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.div(inputs, keep_prob) * binary_tensor
    return output
        


def MBConvBlock(inputs, block_args, is_training, freeze_bn=True, drop_rate=None, prefix='', ):
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = layers.conv2d(inputs, filters, 1,
                          padding='same',
                          use_bias=False,
                          trainable = is_training,
                          kernel_initializer=conv_kernel_initializer,
                          name=prefix + 'expand_conv')
        x = layers.BatchNormalization(x, trainable=freeze_bn, name=prefix + 'expand_bn')
        x = tf.nn.swish(x)
    else:
        x = inputs
    
     # Depthwise Convolution
    x = layers.separable_conv2d(x, tf.shape(x)[-1],
                               block_args.kernel_size,
                               strides=block_args.strides,
                               padding='same',
                               use_bias=False,
                               trainable = is_training,
                               depthwise_initializer=conv_kernel_initializer,
                               name=prefix + 'dwconv')
    x = layers.BatchNormalization(x, trainable=freeze_bn, name=prefix + 'expand_bn')
    x = tf.nn.swish(x)
    
    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(1, int(
            block_args.input_filters * block_args.se_ratio
        ))
        se_tensor = layers.AveragePooling2D(x, tf.shape(x)[1:3], 1, name=prefix + 'se_squeeze')

        target_shape = [-1, 1, 1, filters]#layers.Reshape不包括batch维度
        se_tensor = tf.reshape(se_tensor, target_shape, name=prefix + 'se_reshape')#输出[batch, 1, 1, filters]
        se_tensor = layers.conv2d(se_tensor, num_reduced_filters, 1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  trainable = is_training,
                                  kernel_initializer=conv_kernel_initializer,
                                  name=prefix + 'se_reduce')
        se_tensor = layers.conv2d(se_tensor, filters, 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  trainable = is_training,
                                  kernel_initializer=conv_kernel_initializer,
                                  name=prefix + 'se_expand')
        x = tf.multiply(x, se_tensor, name=prefix + 'se_excite')
    x = layers.conv2d(block_args.output_filters, 1,
                      padding='same',
                      use_bias=False,
                      trainable = is_training,
                      kernel_initializer=conv_kernel_initializer,
                      name=prefix + 'project_conv')
    x = layers.BatchNormalization(x, trainable=freeze_bn, name=prefix + 'project_bn') 
    
    if block_args.id_skip and all(
            s == 1 for s in block_args.strides
    ) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = drop_connect(x, is_training, drop_rate)
        x = layers.add([x, inputs], name=prefix + 'add')
    return x

def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 input_tensor=None,
                 is_training=True,
                 freeze_bn=True, 
                 **kwargs):
    features = []
    # Build stem
    x = input_tensor
    x = layers.conv2d(x, round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      trainable = is_training,
                      kernel_initializer=conv_kernel_initializer,
                      name='stem_conv')
    x = layers.BatchNormalization(x, trainable=freeze_bn, name='stem_bn') 
    x = tf.nn.swish(x)
    
    # Build blocks
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters,
                                        width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters,
                                         width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args,
                          is_training=is_training,
                          freeze_bn = freeze_bn, 
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(
                    idx + 1,
                    string.ascii_lowercase[bidx + 1]
                )
                x = mb_conv_block(x, block_args,
                                  is_training=is_training,
                                  freeze_bn = freeze_bn, 
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1
                
        if idx < len(blocks_args) - 1 and blocks_args[idx + 1].strides[0] == 2:
            features.append(x)
        elif idx == len(blocks_args) - 1:
            features.append(x)
    return features

    

def EfficientNetB0(input_tensor=None,
                   is_training=True,
                   freeze_bn=True,
                   **kwargs):
    return EfficientNet(1.0, 1.0, 224, 0.2,
                        model_name='efficientnet-b0',
                        input_tensor=input_tensor,
                        is_training=True,
                        freeze_bn=True,
                        **kwargs)


def EfficientNetB1(input_tensor=None,
                   is_training=True,
                   freeze_bn=True,
                   **kwargs):
    return EfficientNet(1.0, 1.1, 240, 0.2,
                        model_name='efficientnet-b1',
                        input_tensor=input_tensor,
                        is_training=True,
                        freeze_bn=True,
                        **kwargs)


def EfficientNetB2(input_tensor=None,
                   **kwargs):
    return EfficientNet(1.1, 1.2, 260, 0.3,
                        model_name='efficientnet-b2',
                        input_tensor=input_tensor, 
                        is_training=True,
                        freeze_bn=True,
                        **kwargs)


def EfficientNetB3(input_tensor=None,
                   is_training=True,
                   freeze_bn=True,
                   **kwargs):
    return EfficientNet(1.2, 1.4, 300, 0.3,
                        model_name='efficientnet-b3',
                        input_tensor=input_tensor,
                        is_training=True,
                        freeze_bn=True,
                        **kwargs)


def EfficientNetB4(input_tensor=None,
                   is_training=True,
                   freeze_bn=True,
                   **kwargs):
    return EfficientNet(1.4, 1.8, 380, 0.4,
                        model_name='efficientnet-b4',
                        input_tensor=input_tensor, 
                        is_training=True,
                        freeze_bn=True,
                        **kwargs)


def EfficientNetB5(input_tensor=None,
                   is_training=True,
                   freeze_bn=True,
                   **kwargs):
    return EfficientNet(1.6, 2.2, 456, 0.4,
                        model_name='efficientnet-b5',
                        input_tensor=input_tensor,
                        is_training=True,
                        freeze_bn=True,
                        **kwargs)


def EfficientNetB6(input_tensor=None,
                   is_training=True,
                   freeze_bn=True,
                   **kwargs):
    return EfficientNet(1.8, 2.6, 528, 0.5,
                        model_name='efficientnet-b6',
                        input_tensor=input_tensor,
                        **kwargs)


def EfficientNetB7(input_tensor=None,
                   is_training=True,
                   freeze_bn=True,
                   **kwargs):
    return EfficientNet(2.0, 3.1, 600, 0.5,
                        model_name='efficientnet-b7',
                        input_tensor=input_tensor,
                        is_training=True,
                        freeze_bn=True,
                        **kwargs)

backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
             EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6]
