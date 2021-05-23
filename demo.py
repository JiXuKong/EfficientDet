import tensorflow as tf
import cv2
import copy
from tensorflow.python import pywrap_tensorflow
import numpy as np
import os, sys
import config as cfg 
from model.EfficientDet import Model, DetectHead
from model.timer import Timer


slim = tf.contrib.slim

input_ = tf.placeholder(tf.float32, shape = [1, None, None, 3])
# get_boxes = tf.placeholder(tf.float32, shape = [cfg.batch_size, 80, 4])
# get_classes = tf.placeholder(tf.float32, shape = [cfg.batch_size, 80])


model = Model(cfg.phi, cfg.num_classes, cfg.num_anchors, cfg.weighted_bifpn, False, True)
pred_class, pred_reg, get_shape = model.forward(input_)

anchors = model.get_anchorlist(get_shape, cfg.base_size, cfg.scale, cfg.aspect_ration, cfg.strides)

 _boxes, _cls_scores, _cls_classes = DetectHead(cfg.class_num, cfg.score_threshold, cfg.nms_iou_threshold, cfg.max_detection_boxes_num, cfg.strides).forward([pred_class, pred_reg],anchors)
    
nms_box, nms_score, nms_label = _boxes, _cls_scores, _cls_classes

restore_path = cfg.val_restore_path
g_list = tf.global_variables()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if restore_path is not None:
    print('Restoring weights from: ' + restore_path)
    restorer = tf.train.Saver(g_list)
    restorer.restore(sess, restore_path)
    
if __name__ == '__main__':
    total_timer = Timer()
    save_fpath = r'E:\FCOS\assets'
    imgnm = r'F:\open_dataset\voc07+12\VOCdevkit\test\JPEGImages\000493.jpg'
#     imgnm = './duche5.jpeg'
    img = cv2.imread(imgnm)

    y, x = img.shape[0:2]

    resize_scale_x = x/cfg.image_size
    resize_scale_y = y/cfg.image_size
    img_orig = copy.deepcopy(img)

    img = cv2.resize(img,(cfg.image_size,cfg.image_size))
    img=img[:,:,::-1]
    img=img.astype(np.float32, copy=False)
    mean = np.array([123.68, 116.779, 103.979])
    mean = mean.reshape(1,1,3)
    img = img - mean
    img = np.reshape(img, (1, cfg.image_size, cfg.image_size, 3))
    feed_dict = {
        input_: img
                }
    b, s, l = sess.run([nms_box, nms_score, nms_label], feed_dict = feed_dict)    
    pred_b = b.reshape(-1, 4)
    pred_s = s.reshape(-1,)
    pred_l = l.reshape(-1,)
    plt.figure(figsize=(20,20))
    plt.imshow(np.asarray(img_orig, np.uint8))
    plt.axis('off') 
    current_axis = plt.gca()
    for j in range(pred_b.shape[0]):
        if (pred_s[j]>=0.3):
            print(pred_l[j], pred_s[j])
            x1,y1, x2, y2 = pred_b[j][0]*resize_scale_x, pred_b[j][1]*resize_scale_y, pred_b[j][2]*resize_scale_x, pred_b[j][3]*resize_scale_y
            cls_ = pred_l[j]+1
            cls_name = str(cfg.classes[pred_l[j]+1])
            color = STANDARD_COLORS[cls_]
            current_axis.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color=color, fill=False, linewidth=2))
            current_axis.text(x1, y1, cls_name + str(pred_s[j])[:5], size='x-large', color='white', bbox={'facecolor':'green', 'alpha':0.5})
    plt.savefig(save_path)
    plt.show()