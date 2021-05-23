import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import os, sys
from data.pascal_voc import pascal_voc
import config as cfg 
from model.EfficientDet import Model
from model.loss import LOSS
from model.timer import Timer


data = pascal_voc('train', False, cfg.train_img_path, cfg.train_label_path, cfg.train_img_txt, True)


input_ = tf.placeholder(tf.float32, shape = [cfg.batch_size, cfg.image_size, cfg.image_size, 3])
get_boxes = tf.placeholder(tf.float32, shape = [cfg.batch_size, 80, 5])
get_num = tf.placeholder(tf.float32, shape = [cfg.batch_size,])


model = Model(cfg.phi, cfg.num_classes, cfg.num_anchors, cfg.weighted_bifpn, False, True)
pred_class, pred_reg, get_shape = model.forward(input_)
anchors = model.get_anchorlist(get_shape, cfg.base_size, cfg.scale, cfg.aspect_ration, cfg.strides)

total_loss, regular_loss, loc_loss, cls_loss, normalizer = LOSS([pred_class, pred_reg], [get_boxes, get_num], anchors).forward()

global_step = slim.get_or_create_global_step()
with tf.variable_scope('learning_rate'):
    lr = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(cfg.DECAY_STEP[0]), np.int64(cfg.DECAY_STEP[1])],
                                     values=[cfg.LR, cfg.LR / 10., cfg.LR / 100.])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.MomentumOptimizer(lr, cfg.momentum_rate, use_nesterov=False)
    gradient = optimizer.compute_gradients(total_loss)
    with tf.name_scope('clip_gradients_YJR'):
        gradient = slim.learning.clip_gradient_norms(gradient,cfg.gradient_clip_by_norm)

    with tf.name_scope('apply_gradients'):
        train_op = optimizer.apply_gradients(grads_and_vars=gradient,global_step=global_step)

g_list = tf.global_variables()
save_list = [g for g in g_list if ('Momentum' not in g.name)and('ExponentialMovingAverage' not in g.name)]
saver = tf.train.Saver(var_list=save_list, max_to_keep=30)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(cfg.ckecpoint_file, sess.graph)


def get_variables_in_checkpoint_file(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return var_to_shape_map

def initialize(pretrained_model, variable_to_restore):
    var_keep_dic = get_variables_in_checkpoint_file(pretrained_model)
    # Get the variables to restore, ignoring the variables to fix
    variables_to_restore = get_variables_to_restore(variable_to_restore, var_keep_dic)
    restorer = tf.train.Saver(variables_to_restore)
    return restorer

def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
        if (v.name == 'global_step:0'):
            continue;

#         if(v.name.split('/')[1] != 'ClassPredictor')\
#         and(v.name.split('/')[1] != 'BoxPredictor')\
#         and(v.name.split(':')[0])in var_keep_dic:
        if (v.name.split(':')[0])in var_keep_dic:

            print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)
        
    return variables_to_restore

if cfg.train_restore_path is not None:
    print('Restoring weights from: ' + cfg.train_restore_path)
    restorer = initialize(cfg.train_restore_path, g_list)
#     restorer = tf.train.Saver(save_list)
    restorer.restore(sess, cfg.train_restore_path)
    

def train():
    total_timer = Timer()
    train_timer = Timer()
    load_timer = Timer()
    max_epoch = 50
    epoch_step = int(cfg.train_num//cfg.batch_size)
    t = 1
    for epoch in range(1, max_epoch + 1):
        print('-'*25, 'epoch', epoch,'/',str(max_epoch), '-'*25)


        t_loss = 0
        ll_loss = 0
        r_loss = 0
        c_loss = 0
        
        
       
        for step in range(1, epoch_step + 1):
     
            t = t + 1
            total_timer.tic()
            load_timer.tic()
 
            images, labels, imnm, num_boxes, imsize = data.get()
            
#             load_timer.toc()
            feed_dict = {input_: images,
                         get_boxes: labels,
                         get_num: num_boxes
                        }

            _, g_step_, total_loss, l2_loss, rloss_, closs_, p_nm_, lr = sess.run(
                [train_op,
                 global_step,
                 total_loss,
                 regular_loss, 
                 loc_loss, 
                 cls_loss,
                 normalizer,learning_rate], feed_dict = feed_dict)
            
            
            total_timer.toc()
            if g_step_%50 ==0:
                sys.stdout.write('\r>> ' + 'iters '+str(g_step_)+str('/')+str(epoch_step*max_epoch)+' loss '+str(tt_loss) + ' ')
                sys.stdout.flush()

                train_total_summary = tf.Summary(value=[
                    tf.Summary.Value(tag="config/learning rate", simple_value=lr),
                    tf.Summary.Value(tag="train/classification/focal_loss", simple_value=cfg.class_weight*closs_),
                    tf.Summary.Value(tag="train/p_nm", simple_value=p_nm_),
                    tf.Summary.Value(tag="train/regress_loss", simple_value=cfg.regress_weight*rloss_),
                    tf.Summary.Value(tag="train/clone_loss", simple_value=cfg.class_weight*closs_ + cfg.regress_weight*rloss_),
                    tf.Summary.Value(tag="train/l2_loss", simple_value=l2_loss),
                    tf.Summary.Value(tag="train/total_loss", simple_value=total_loss)
                    ])
                print('curent speed: ', total_timer.diff, 'remain time: ', total_timer.remain(g_step_, epoch_step*max_epoch))
                summary_writer.add_summary(train_total_summary, g_step_)
            if g_step_%10000 == 0:
                print('saving checkpoint')
                saver.save(sess, cfg.ckecpoint_file + '/model.ckpt', g_step_)

        total_timer.toc()
        sys.stdout.write('\n')
        print('>> mean loss', t_loss)
        print('curent speed: ', total_timer.average_time, 'remain time: ', total_timer.remain(g_step_, epoch_step*max_epoch))
        
    print('saving checkpoint')
    saver.save(sess, cfg.ckecpoint_file + '/model.ckpt', g_step_)
    
    
train()    
    
    
    
    
    