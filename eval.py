import tensorflow as tf
import cv2
import copy
from tensorflow.python import pywrap_tensorflow
import numpy as np
import os, sys
from data.pascal_voc import pascal_voc
import config as cfg 
from model.EfficientDet import Model, DetectHead
from model.loss import LOSS
from evalue import voc_eval
from model.timer import Timer


slim = tf.contrib.slim

input_ = tf.placeholder(tf.float32, shape = [1, None, None, 3])
get_boxes = tf.placeholder(tf.float32, shape = [cfg.batch_size, 80, 5])
get_num = tf.placeholder(tf.float32, shape = [cfg.batch_size, ])


model = Model(cfg.phi, cfg.num_classes, cfg.num_anchors, cfg.weighted_bifpn, False, True)
pred_class, pred_reg, get_shape = model.forward(input_)

anchors = model.get_anchorlist(get_shape, cfg.base_size, cfg.scale, cfg.aspect_ration, cfg.strides)

 _boxes, _cls_scores, _cls_classes = DetectHead(cfg.class_num, cfg.score_threshold, cfg.nms_iou_threshold, cfg.max_detection_boxes_num, cfg.strides).forward([pred_class, pred_reg],anchors)
    
targets = loss.GenTargets(cfg.strides, cfg.limit_range).forward((out, get_boxes, get_classes))

total_loss, _, loc_loss, cls_loss, _ = LOSS([pred_class, pred_reg], [get_boxes, get_num], anchors).forward()

nms_box, nms_score, nms_label = _boxes, _cls_scores, _cls_classes
restore_path = cfg.val_restore_path
epoch = int(restore_path.split('-')[-1])//2000
print('epoch', epoch)


g_list = tf.global_variables()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(ckecpoint_file, sess.graph)


if restore_path is not None:
    print('Restoring weights from: ' + restore_path)
    restorer = tf.train.Saver(g_list)
    restorer.restore(sess, restore_path)
    
if __name__ == '__main__':

    val_timer = Timer()
    data = pascal_voc('test', False, cfg.test_img_path, cfg.test_label_path, cfg.test_img_txt, True)
    val_pred = []
    gt_dict = {}
    val_rloss = 0
    val_closs = 0
    val_clone_loss = 0
    for val_step in range(1, cfg.test_num+1):
        val_timer.tic()
        images, labels, imnm, num_boxes, imsize = data.get()
        feed_dict = {input_: images,
                         get_boxes: labels,
                         get_num: num_boxes
                        }
        b, s, l, valrloss_, valcloss_, valtotal_loss = sess.run([nms_box, nms_score, nms_label, loc_loss, cls_loss, total_loss],
                                                      feed_dict = feed_dict)
        val_rloss += valrloss_/cfg.test_num
        val_closs += valcloss_/cfg.test_num
        val_clone_loss += (valrloss_ + valcloss_)/cfg.test_num
        
        for i in range(cfg.batch_size):
            pred_b = b[i]
            pred_s = s[i]
            pred_l = l[i]
            for j in range(pred_b.shape[0]):
                if pred_l[j] >=0 :
                    val_pred.append([imnm[i], pred_b[j][0], pred_b[j][1], pred_b[j][2], pred_b[j][3], pred_s[j], pred_l[j]+1])
            single_gt_num = np.where(labels[i][:,0]>0)[0].shape[0]
            box = np.hstack((labels[i][:single_gt_num, 1:], np.reshape(labels[i][:single_gt_num, 0], (-1,1)))).tolist()
            gt_dict[imnm[i]] = box 

        val_timer.toc()    
        sys.stdout.write('\r>> ' + 'val_nums '+str(val_step)+str('/')+str(cfg.test_num+1))
        sys.stdout.flush()

    print('curent val speed: ', val_timer.average_time, 'val remain time: ', val_timer.remain(val_step, cfg.test_num+1))
    print('val mean regress loss: ', val_rloss, 'val mean class loss: ', val_closs, 'val mean total loss: ', val_clone_loss)
    mean_rec = 0
    mean_prec = 0
    mAP = 0
    for classidx in range(1, cfg.class_num):#从1到21，对应[bg,...]21个类（除bg）
        rec, prec, ap = voc_eval(gt_dict, val_pred, classidx, iou_thres=0.5, use_07_metric=False)

        print(cfg.classes[classidx] + ' ap: ', ap)
        mean_rec += rec[-1]/(cfg.class_num-1)
        mean_prec += prec[-1]/(cfg.class_num-1)
        mAP += ap/(cfg.class_num-1)
    print('Epoch: ' + str(epoch), 'mAP: ', mAP)
    print('Epoch: ' + str(epoch), 'mRecall: ', mean_rec)
    print('Epoch: ' + str(epoch), 'mPrecision: ', mean_prec)
    
    
    val_total_summary2 = tf.Summary(value=[
        tf.Summary.Value(tag="val/loss/class_loss", simple_value=val_closs),
        tf.Summary.Value(tag="val/loss/regress_loss", simple_value=val_rloss),
        tf.Summary.Value(tag="val/loss/clone_loss", simple_value=val_clone_loss),
        tf.Summary.Value(tag="val/mA", simple_value=mAP),
        tf.Summary.Value(tag="val/mRecall", simple_value=mean_rec),
        tf.Summary.Value(tag="val/mPrecision", simple_value=mean_prec),
     ])
    
    summary_writer.add_summary(val_total_summary2, epoch)