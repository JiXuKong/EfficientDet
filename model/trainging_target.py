import numpy as np
import tensorflow as tf
import numpy.random as npr
from xx import iou_tf, regress_target_tf


def get_training_targets(
        anchors, groundtruth_boxes, groundtruth_labels, img_size,
        positives_threshold=0.5, negatives_threshold=0.4):

    with tf.name_scope('matching'):

        N = tf.shape(groundtruth_boxes)[0]
        num_anchors = tf.shape(anchors)[0]
        only_background = tf.fill([num_anchors], -1)

        matches = tf.to_int32(tf.cond(
            tf.greater(N, 0),
            lambda: match_boxes(
                anchors, groundtruth_boxes,
                positives_threshold=positives_threshold,
                negatives_threshold=negatives_threshold,
                force_match_groundtruth=True
            ),
            lambda: only_background
        ))

    with tf.name_scope('target_creation'):
        reg_targets, cls_targets = create_targets(
            anchors, groundtruth_boxes,
            groundtruth_labels, matches, img_size
        )

    return reg_targets, cls_targets, matches


def match_boxes(
        anchors, groundtruth_boxes, positives_threshold=0.5,
        negatives_threshold=0.4, force_match_groundtruth=True):

    assert positives_threshold >= negatives_threshold
    # for each anchor box choose the groundtruth box with largest iou
    similarity_matrix = iou_tf(anchors, groundtruth_boxes)  # shape [N, num_anchors]
    matches = tf.argmax(similarity_matrix, axis=0, output_type=tf.int32)  # shape [num_anchors]#对每个anchor找一个gt
    matched_vals = tf.reduce_max(similarity_matrix, axis=0)  # shape [num_anchors]#求得对应的gt-anchor的iou
    is_positive = tf.to_int32(tf.greater_equal(matched_vals, positives_threshold))#tf.greater_equal：所有大于阈值iou的索引位置=1，其余为0

    if positives_threshold == negatives_threshold:#如果没有忽略的anchor
        is_negative = 1 - is_positive#负的
        matches = matches * is_positive + (-1 * is_negative)#matches[i]:>0代表该anchori匹配到了对应的gt
    else:
        is_negative = tf.to_int32(tf.greater(negatives_threshold, matched_vals))#如果有忽略的anchor，对应anchor位置=0
        to_ignore = (1 - is_positive) * (1 - is_negative)#忽略的
        matches = matches * is_positive + (-1 * is_negative) + (-2 * to_ignore)

    # after this, it could happen that some groundtruth
    # boxes are not matched with any anchor box

    if force_match_groundtruth:
        # now we must ensure that each row (groundtruth box) is matched to
        # at least one column (which is not guaranteed
        # otherwise if `positives_threshold` is high)

        # for each groundtruth box choose the anchor box with largest iou
        # (force match for each groundtruth box)
        forced_matches_ids = tf.argmax(similarity_matrix, axis=1, output_type=tf.int32)  # shape [N]
        # if all indices in forced_matches_ids are different then all rows will be matched

        num_anchors = tf.shape(anchors)[0]
        forced_matches_indicators = tf.one_hot(forced_matches_ids, depth=num_anchors, dtype=tf.int32)  # shape [N, num_anchors]
        forced_match_row_ids = tf.argmax(forced_matches_indicators, axis=0, output_type=tf.int32)  # shape [num_anchors]

        # some forced matches could be very bad!
        forced_matches_values = tf.reduce_max(similarity_matrix, axis=1)  # shape [N]
        small_iou = 0.1  # this requires that forced match has at least small intersection
        is_okay = tf.to_int32(tf.greater_equal(forced_matches_values, small_iou))  # shape [N]
        forced_matches_indicators = forced_matches_indicators * tf.expand_dims(is_okay, axis=1)

        forced_match_mask = tf.greater(tf.reduce_max(forced_matches_indicators, axis=0), 0)  # shape [num_anchors]
        matches = tf.where(forced_match_mask, forced_match_row_ids, matches)
        # even after this it could happen that some rows aren't matched,
        # but i believe that this event has low probability
    
    
    
    
    return matches



def create_targets(anchors, groundtruth_boxes, groundtruth_labels, matches, img_size):
    matched_anchor_indices = tf.where(tf.greater_equal(matches, 0))  # shape [num_matches, 1]
    matched_anchor_indices = tf.to_int32(tf.squeeze(matched_anchor_indices, axis=1))

    unmatched_anchor_indices = tf.where(tf.less(matches, 0))  # shape [num_anchors - num_matches, 1]
    unmatched_anchor_indices = tf.to_int32(tf.squeeze(unmatched_anchor_indices, axis=1))

    matched_gt_indices = tf.gather(matches, matched_anchor_indices)  # shape [num_matches]
    matched_gt_boxes = tf.gather(groundtruth_boxes, matched_gt_indices)  # shape [num_matches, 4]
    matched_anchors = tf.gather(anchors, matched_anchor_indices)  # shape [num_matches, 4]

    matched_reg_targets = regress_target_tf(matched_gt_boxes, matched_anchors, img_size)  # shape [num_matches, 4]
    matched_cls_targets = tf.gather(groundtruth_labels, matched_gt_indices)  # shape [num_matches]
    matched_cls_targets = matched_cls_targets  # background class will have index `0`#不需要+1,因为我们的标签从1开始1-21

    num_unmatched = tf.size(unmatched_anchor_indices)  # num_anchors - num_matches
    unmatched_reg_targets = tf.zeros([num_unmatched, 4], dtype=tf.float32)
    unmatched_cls_targets = tf.zeros([num_unmatched], dtype=tf.int32)

    reg_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_reg_targets, unmatched_reg_targets]
    )  # shape [num_anchors, 4]

    cls_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_cls_targets, unmatched_cls_targets]
    )  # shape [num_anchors]    
    
    return reg_targets, cls_targets