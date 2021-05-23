import math
import numpy as np
import tensorflow as tf
import tensorflow.layers as layers

'''
ClsCntRegHead类：两个预测分支，返回分支列表
'''
use_dw=False
if use_dw:
    w_i_1 = tf.variance_scaling_initializer()
    w_i_2 = tf.variance_scaling_initializer()
else:
    w_i_1 = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    w_i_2 = None
b_i = tf.constant_initializer(0.0)

def conv_op(x, c, k, trainable, strides, padding="SAME", use_bias=True, scope=None, reuse=False, weights_initializer_1=w_i_1, weights_initializer_2=w_i_2, bias_initializer=b_i, separable_conv=use_dw):
    if separable_conv:
        x = layers.separable_conv2d(x, c,
                                    k,
                                    strides=strides,
                                    padding=padding,
                                    use_bias=use_bias,
                                    trainable = trainable,
                                    depthwise_initializer=weights_initializer_1,
                                    pointwise_initializer=weights_initializer_2,
                                    bias_initializer=bias_initializer,
                                    name=scope)
    else:
        x = layers.conv2d(x, c,
                          k,
                          strides=strides,
                          padding=padding,
                          use_bias=use_bias,
                          trainable = trainable,
                          kernel_initializer=weights_initializer_1,
                          bias_initializer=bias_initializer,
                          name=scope)
    return x

class ClsRegHead(object):
    def __init__(self, width, depth, num_anchors, class_num, is_training, prior=0.01):
        self.is_training = is_training
        self.prior=prior
        self.class_num=class_num
        self.out_channel = width
        self.depth = depth
          
    #subnet权重共享
    def baseclassification_subnet(self, features, feature_level):
        reuse1 = tf.AUTO_REUSE
        for j in range(self.depth):
            features = conv_op(features, self.out_channel, [3,3], trainable=self.is_training,
                               strides=1,
                               padding="SAME",
                               scope='ClassPredictionTower/conv2d_' + str(j),# + str(feature_level), 
                               reuse=reuse1)

            features = tf.layers.group_norm(features, trainable=self.is_training, scope= 'ClassPredictionTower/conv2d_%d/GroupNorm/feature_%d' % (j, feature_level))
            features = tf.nn.relu(features)

        class_feature_output = conv_op(features, (self.class_num-1)* num_anchors, [3,3], trainable=self.is_training,
                                   biases_initializer=tf.constant_initializer(-math.log((1 - self.prior)/self.prior)),
                                   strides=1,
                                   scope='ClassPredictor', 
                                   reuse=reuse1)
        
        

        return class_feature_output
    def baseregression_subnet(self, features, feature_level):    
        reuse2 = tf.AUTO_REUSE
        for j in range(self.depth):
            features = conv_op(features, self.out_channel, [3,3], trainable=self.is_training,
                                   strides=1,
                                   padding="SAME",
                                   scope='BoxPredictionTower/conv2d_' + str(j),# + str(feature_level), 
                                   reuse=reuse2)
            features = tf.contrib.layers.group_norm(features, trainable=self.is_training, scope='BoxPredictionTower/conv2d_%d/GroupNorm/feature_%d' % (j, feature_level))
            features = tf.nn.relu(features)
        regress_feature_output = conv_op(features, 4, [3,3], trainable=self.is_training,
                                   stride=1,
                                   scope='BoxPredictor', 
                                   reuse=reuse2)

        return regress_feature_output
    
    def pred_subnet(self, fpn_features):
        cfeatures_ = []
        rfeatures_ = []
        get_shape = []
        with tf.variable_scope('WeightSharedConvolutionalBoxPredictor'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(1e-4)):
                for i in range(3, len(fpn_features)+3):
                    class_ = self.baseclassification_subnet(fpn_features[i-3], i-3)
                    cfeatures_.append(tf.reshape(class_, [tf.sahpe(class_)[0], -1, tf.sahpe(class_)[-1]]))
                    get_shape.append(tf.sahpe(class_)[1:2])

                    box_ = self.baseregression_subnet(fpn_features[i-3], i-3)
                    rfeatures_.append(tf.reshape(box_, [tf.sahpe(box_)[0], -1, tf.sahpe(box_)[-1]])) 

                return tf.concat(cfeatures_, axis = 1), tf.concat(rfeatures_, axis = 1), get_shape
    
        
    