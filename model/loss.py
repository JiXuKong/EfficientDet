import tensorflow as tf
import numpy as np

from model.trainging_target import get_training_targets

gama = 2.0
alpha = 0.25
std=[5.0, 5.0, 5.0, 5.0]

class LOSS(pred, gt, anchors, class_weight = 2.0， regress_weight = 1.0):
    '''
    pred[0]:[batch, num, 20]
    pred[1]:[batch, num, 4*9]
    gt[0]:[batch, 80, 5], <l,x1,y1,x2,y2>
    gt[1]: [batch, ]
    anchors:[num, 4]
    '''
    def __init__(self):
        self.pred_c = pred[0]
        self.pred_b = pred[1]
        self.labels = gt[0] 
        self.num_boxes = gt[1]
        self.anchors = anchors
        self.class_weight = class_weight
        self.regress_weight = regress_weight
    def batch_target(self):
        def fn(x):
            boxes, labels, num_boxes = x
            boxes, labels = boxes[:num_boxes], labels[:num_boxes]

            reg_targets, cls_targets, matches = get_training_targets(
                self.anchor, boxes, labels, self.img_size,
                positives_threshold=0.5,
                negatives_threshold=0.4
            )
            return reg_targets, cls_targets, matches

        with tf.name_scope('target_creation'):
            reg_targets, cls_targets, matches = tf.map_fn(
                fn, [self.label[:, :, 1:], self.label[:, :, 0], self.num_boxes],
                dtype=(tf.float32, tf.int32, tf.int32),
                parallel_iterations=4,
                back_prop=False, swap_memory=False, infer_shape=True
            )
            return reg_targets, cls_targets, matches
    def forward(self):
        reg_targets, cls_targets, matches = self.batch_target()
        with tf.name_scope('losses'):
            weights = tf.to_float(tf.greater_equal(matches, 0))
            with tf.name_scope('classification_loss'):
                cls_targets = tf.one_hot(cls_targets, self.class_num, axis=2)
                cls_targets = tf.to_float(cls_targets[:, :, 1:])
                not_ignore = tf.to_float(tf.greater_equal(matches, -1))
                cls_losses = focal_loss(
                    self.pred_c, cls_targets, weights=not_ignore,
                    gamma=gama, alpha=alpha)
            with tf.name_scope('localization_loss'):
                encoded_boxes = tf.identity(self.pred_regress_target_list)
                loc_losses = localization_loss(encoded_boxes, reg_targets, weights)
            with tf.name_scope('normalization'):
                matches_per_image = tf.reduce_sum(weights, axis=1)  # shape [batch_size]
                num_matches = tf.reduce_sum(matches_per_image, axis=0)  # shape []
                normalizer = tf.maximum(num_matches, 1.0)    
        
            loc_loss = tf.reduce_sum(loc_losses, axis=[0, 1])/normalizer
            cls_loss = tf.reduce_sum(cls_losses, axis=[0, 1])/normalizer
            tf.losses.add_loss(self.class_weight*cls_loss)
            tf.losses.add_loss(self.regress_weight*loc_loss)
            with tf.name_scope('weight_decay'):
                slim_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                regularization_loss = tf.losses.get_regularization_loss()            
            total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
            
        return total_loss, regularization_loss, loc_loss, cls_loss, normalizer

def regress_target_tf(gt_box, anchor_box, img_size, box_std=std):
    '''
    gt_box: [M, 4]
    x1, y1, x2, y2
    anchor_box:[M, 4]
    x1, y1, x2, y2
    img_size: [h, w]
    '''
    gt_box = tf.to_float(gt_box)
    anchor_box = tf.to_float(anchor_box)
    x1, y1, x2, y2 = tf.split(gt_box, [1,1,1,1], -1)
    x1, y1, x2, y2 = x1/img_size[1], y1/img_size[0], x2/img_size[1], y2/img_size[0]
    w = x2 - x1
    h = y2 - y1
    x_ctr = x1 + w/2
    y_ctr = y1 + h/2
    
    xx1, yy1, xx2, yy2 = tf.split(anchor_box, [1,1,1,1], -1)
    xx1, yy1, xx2, yy2 = xx1/img_size[1], yy1/img_size[0], xx2/img_size[1], yy2/img_size[0]
    ww = xx2 - xx1
    hh = yy2 - yy1
    xx_ctr = xx1 + ww/2
    yy_ctr = yy1 + hh/2
    
    tx = (x_ctr-xx_ctr)/ww
    ty = (y_ctr-yy_ctr)/hh
    tw = tf.log(w/ww)
    th = tf.log(h/hh)
    return tf.concat([tx*box_std[0],ty*box_std[1],tw*box_std[2],th*box_std[3]], axis = -1)

def reverse_regress_target_tf(pred_box, anchor_box, img_size, std=std):
    '''
    anchor_box: [-1, 4]
    pred_box:[-1, 4]
    '''
    anchor_box = tf.to_float(anchor_box)
    xx1, yy1, xx2, yy2 = tf.split(anchor_box, [1,1,1,1], -1)
    xx1, yy1, xx2, yy2 = xx1/img_size[1], yy1/img_size[0], xx2/img_size[1], yy2/img_size[0]
    ww = xx2 - xx1
    hh = yy2 - yy1
    xx_ctr = xx1 + ww/2.0
    yy_ctr = yy1 + hh/2.0
    
    tx, ty, tw, th = tf.split(pred_box, [1,1,1,1], -1)
    tx, ty, tw, th = tx/box_std[0], ty/box_std[1], tw/box_std[2], th/box_std[3]
    etw, eth = tf.exp(tw), tf.exp(th)
    h, w = hh*eth, ww*etw
    x_ctr, y_ctr = tx*ww + xx_ctr, ty*hh + yy_ctr
    x1 = x_ctr - w/2.0
    y1 = y_ctr - h/2.0
    x2 = x_ctr + w/2.0
    y2 = y_ctr + h/2.0
    return tf.stop_gradient(tf.concat([x1*img_size[1],y1*img_size[0],x2*img_size[1],y2*img_size[0]], axis=-1))

    
    
    

def localization_loss(predictions, targets, weights):
    """A usual L1 smooth loss.

    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, 4],
            representing the (encoded) predicted locations of objects.
        targets: a float tensor with shape [batch_size, num_anchors, 4],
            representing the regression targets.
        weights: a float tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    abs_diff = tf.abs(predictions - targets)
    abs_diff_lt_1 = tf.less(abs_diff, 1.0)
    loss = tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
    return weights * tf.reduce_sum(loss, axis=2)


def focal_loss(predictions, targets, weights, gamma=2.0, alpha=0.25):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, num_classes],
            representing the predicted logits for each class.
        targets: a float tensor with shape [batch_size, num_anchors, num_classes],
            representing one-hot encoded classification targets.
        weights: a float tensor with shape [batch_size, num_anchors].
        gamma, alpha: float numbers.
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    positive_label_mask = tf.equal(targets, 1.0)

#     delta = 0.01
#     targets = (1 - delta) * targets + + delta * 1. / 2
    negative_log_p_t = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=predictions)
#     negative_log_p_t = tf.losses.sigmoid_cross_entropy(multi_class_labels=targets, logits = predictions, label_smoothing=0)
    
    
    probabilities = tf.sigmoid(predictions)
    p_t = tf.where(positive_label_mask, probabilities, 1.0 - probabilities)
    # they all have shape [batch_size, num_anchors, num_classes]

    modulating_factor = tf.pow(1.0 - p_t, gamma)
    weighted_loss = tf.where(
        positive_label_mask,
        alpha * negative_log_p_t,
        (1.0 - alpha) * negative_log_p_t
    )
    focal_loss = modulating_factor * weighted_loss
    # they all have shape [batch_size, num_anchors, num_classes]

    return weights * tf.reduce_sum(focal_loss, axis=2)

