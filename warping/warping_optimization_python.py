#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:03:37 2019

@author: emre
"""

#import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
#from WarpingHCI import WarpingHCI, WarpingHCI_SOFT, WarpingHCI_SOFT2, computeMSE
from WarpingHCI import Warping_MultiTarget_CRIT, computeMSE_MultiTarget
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
step_size = 0.001
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

mu1=1
mu0=0

for itr in range(max_iters):
    SoftWarpeds,CRITs = Warping_MultiTarget_CRIT(LF, Dref, target_views, ref_view,mu1=mu1,mu0=mu0)
    total_MSE = computeMSE_MultiTarget(SoftWarpeds,LF,target_views)
    print (total_MSE)
    
#    dMSE_over_dmu1, dMSE_over_dmu0 = compute_gradient(CRITs)
#    mu1 = mu1 - step_size*dMSE_over_dmu1
#    mu0 = mu0 - step_size*dMSE_over_dmu0

#for itr in range(max_iters):










