import math
import numpy as np
import tensorflow as tf
import tensorflow.layers as layers
from model.EfficientNet.efficientnet import backbones
from model.BiFPN_Neck import build_wBiFPN
from model.Head import ClsRegHead
import reverse_regress_target_tf
import gpu_nms

w_bifpns = [64, 88, 112, 160, 224, 288, 384]
d_bifpns = [3, 4, 5, 6, 7, 7, 8]
d_heads = [3, 3, 3, 4, 4, 4, 5]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
batch_size = 4


def generate_anchor_(base_size, scale, aspect_ration, feature_size, stride):
    area = base_size*base_size*scale**2
    w = np.sqrt(area.reshape((area.shape[0],1))/aspect_ratio.reshape(1,aspect_ratio.shape[0]))
    h = aspect_ratio*w
    w = w.transpose()
    h = h.transpose()
    w = w.reshape(-1)
    h = h.reshape(-1)
    base_anchor = np.vstack((-w/2, -h/2, w/2, h/2)).transpose()
    grid = np.array([[j*stride[0]+stride[0]/2, i*stride[1]+stride[1]/2, j*stride[0]+stride[0]/2, i*stride[1]+stride[1]/2] for i in range(feature_size[1]) for j in range(feature_size[0])])
    generate_anchor = grid.reshape((-1, 1 ,4)) + base_anchor.reshape(1, -1, 4)
    generate_anchor = generate_anchor.reshape(-1, 4)    
    return generate_anchor

class Model(object):
    def __init__(self, phi, num_classes=20, num_anchors=9, weighted_bifpn=False, freeze_bn=False, is_training=True):
        assert phi in range(7)
        self.is_training = is_training
        self.phi = phi
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.weighted_bifpn = weighted_bifpn
        self.freeze_bn = freeze_bn
        self.is_training = is_training
        
    def forward(self, input_):
        w_bifpn = w_bifpns[self.phi]
        d_bifpn = d_bifpns[self.phi]
        w_head = w_bifpn
        d_head = d_heads[self.phi]
        backbone_cls = backbones[self.phi]
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(4e-4)):
            features = backbone_cls(input_tensor=input_, freeze_bn=self.freeze_bn, is_training=self.is_training)
            fpn_features = features
            for i in range(d_bifpn):
                fpn_features = build_wBiFPN(fpn_features, w_bifpn, i, self.weighted_bifpn, self.is_training, self.freeze_bn)
            pred_class, pred_reg，get_shape = ClsRegHead(w_head, d_head, self.num_anchors, self.num_classes, self.is_training).pred_subnet(fpn_features)
            return pred_class, pred_reg, get_shape
    
    def get_anchorlist(self, get_shape, base_size, scale, aspect_ration, strides):
        anchorlist = []
        for i in range(len(get_shape)):
            feature_size = get_shape[i]
            strides = stride
            anchor = tf.py_func(generate_anchor_,
                      [base_size, scale, aspect_ration, feature_size, stride],
                      [tf.float32])
            anchorlist.append(anchor)
        return tf.concat(anchorlist, axis = 0)
        

    
class DetectHead(object):
    def __init__(self, class_num, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides):
        self.class_num = class_num
        self.score_threshold=score_threshold
        self.nms_iou_threshold=nms_iou_threshold
        self.max_detection_boxes_num=max_detection_boxes_num
        self.strides=strides

    def forward(self, inputs, anchors):
        '''
        inputs[0]: [batch_size,h*w, class_num]
        inputs[1]: [batch_size,h*w, anchor_nums*4]
        anchors:[h*w, 4]
        '''
        pred_class, pred_reg = inputs
        pred_logits = tf.nn.sigmoid(pred_class)
        batch_nms_box = []
        batch_nms_score = []
        batch_nms_label = []
        for i in range(batch_size):
            pred_logits_i = pred_logits[i]
            pred_reg_i = pred_reg[i]
            box_i = reverse_regress_target_tf(pred_reg_i, anchors)
            nms_box, nms_score, nms_label = gpu_nms(box_i, pred_logits_i, self.class_num-1, 
                                                    self.max_detection_boxes_num, 
                                                    self.score_threshold, 
                                                    self.nms_iou_threshold)
            batch_nms_box.append(nms_box)
            batch_nms_score.append(nms_score)
            batch_nms_label.append(nms_label)
        return batch_nms_box, batch_nms_score, batch_nms_label
        
        
        
