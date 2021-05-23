import numpy as np

###data path
train_img_txt = r'F:\open_dataset\VOCdevkit\0712train\07train.txt'
train_img_path = r'F:\open_dataset\VOCdevkit\0712train\JPEGImages'
train_label_path = r'F:\open_dataset\VOCdevkit\0712train\Annotations'
train_num = 50000

test_img_txt = r'F:\open_dataset\VOCdevkit\07test\test.txt'
test_img_path = r'F:\open_dataset\VOCdevkit\07test\JPEGImages'
test_label_path = r'F:\open_dataset\VOCdevkit\07test\Annotations'
test_num = 4957
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

##save/restore path
cache_path = './pkl'
val_restore_path =  './checkpoint/model.ckpt-62499'
train_restore_path = r'E:\RetinaNet_copy\pretrain_weight\resnet_v1_50_2016_08_28\resnet_v1_50.ckpt'
# train_restore_path = './checkpoint/model.ckpt-99999'
ckecpoint_file = './checkpoint'


#data aug
gridmask = False
random_crop = False
other_aug = False
multiscale = False
class_to_ind = 0

weight_decay = 0.0001
momentum_rate = 0.9
gradient_clip_by_norm = 10.0

strides = [8, 16, 32, 64, 128]
base_anchor = [32, 64, 128, 256, 512]
scale = np.array([1, 2**(1/2)])
aspect_ratio = np.array([1.0, 2.0, 0.5])
score_threshold=0.05
nms_iou_threshold=0.6
max_detection_boxes_num = 1000

batch_size = 1
LR = 0.01/batch_size
DECAY_STEP = [train_num//batch_size*40, train_num//batch_size*47]
class_weight = 1.
regress_weight = 1.
cnt_weight = 1.

class_num = len(classes)
image_size = 1024