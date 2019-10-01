#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:03:37 2019

@author: emre
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from WarpingHCI import WarpingHCI_SOFT2, computeMSE, tf_Warping_MultiTarget, tf_computeMSE_MultiTarget
#from tfWarping import Warping_MultiTarget, computeMSE_MultiTarget
import sys, os
sys.path.append("/home/emre/Documents/kodlar/")
sys.path.append("/home/emre/Documents/kodlar/epinet-master/epinet_fun")
from func_pfm import read_pfm
from datasets import datasets
from shutil import copyfile
from target_patterns import window
from target_patterns import visualize as vis_tv
import inspect
import time
import cv2
#from scipy.io import savemat
import h5py
from warping_usefuls import pad_D
###################
#### Config ####

#ds = datasets.hci("/media/emre/Data/heidelberg_full_data/")
ds = datasets.lytro("/media/emre/Data/Lytro_lenslet/")
sample = ds.samples[0]
#Warping_Function = WarpingHCI_SOFT2
#############################
max_iters = 1000
ref_view = (7,7)
crop_LF = True
target_pattern = window
pfm_dir = '/media/emre/Data/epinet_tests/25-02-11-24__lytro/pfms/'
##############################

#################
disp_file = pfm_dir + sample + "_" + str(ref_view[0]) + "_" + str(ref_view[1]) + ".pfm"
Dref = read_pfm(disp_file)
if crop_LF:
    crop_size = Dref.shape
else:
    crop_size = None
    Dref = pad_D(Dref,ds.im_size)



LF = ds.readLF(sample, vis = True, crop_size = crop_size)
LFref = LF[ref_view]

target_views = target_pattern(ref_view)

vis_tv(ds.iar_max, ds.iac_max, target_views)


mu1 = tf.Variable(initial_value=1,dtype=tf.float32)
mu0 = tf.Variable(initial_value=0,dtype=tf.float32)
D = mu1*Dref + mu0    


#SoftWarped = tf.py_func(WarpingHCI_SOFT2,inp,[tf.float32])
SoftWarpeds = tf_Warping_MultiTarget(LFref, D, target_views, ref_view)

total_MSE = tf.py_func(tf_computeMSE_MultiTarget, [SoftWarpeds,LF,target_views], tf.float32)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
opt = optimizer.minimize(total_MSE)
#opt = optimizer.minimize(SoftWarpeds)
#
sess = tf.Session()
#
for itr in range(max_iters):
    _,vMSE,vmu1,vmu0 = sess.run([opt,total_MSE,mu1,mu0])
    print(vMSE)
    print(vmu1)
    print(vmu0)






