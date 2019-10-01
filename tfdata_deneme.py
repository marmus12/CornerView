# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 16:36:12 2019

@author: vhemka
"""
import numpy as np
import tensorflow as tf
from epinet_fun.util import load_LFdata, load_depth_gts


traindir_LFimages=['additional\\antinous', 'additional\\boardgames']
hci_root = "D:\\heidelberg_full_data\\"        
add_gt_dir = "D:\\heidelberg_full_data\\additional_depth_disp_all_views\\"

# Load the training data into two NumPy arrays, for example using `np.load()`.
    
train_ims,_=load_LFdata(traindir_LFimages,hci_root)
train_labels = load_depth_gts(add_gt_dir, traindir_LFimages)#[:,:,:,label_view_ind] 


sess = tf.Session()

# Assume that each row of `features` corresponds to the same row as `labels`.
assert train_ims.shape[0] == train_labels.shape[0]

features_placeholder = tf.placeholder(train_ims.dtype, train_ims.shape)
labels_placeholder = tf.placeholder(train_labels.dtype, train_labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
#dataset = ...
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: train_ims,
                                          labels_placeholder: train_labels})
