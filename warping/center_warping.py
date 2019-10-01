#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:34:58 2019

@author: emre
"""

import numpy as np
from matplotlib import pyplot as plt
from warping_lib import  WarpingHCI_SOFT2,  WarpingHCI_SOFT3, computeMSE
import sys, os
sys.path.append("/home/emre/Documents/kodlar/")
sys.path.append("/home/emre/Documents/kodlar/epinet-master/epinet_fun")
from func_pfm import read_pfm
from datasets import datasets
from shutil import copyfile
from warping_usefuls import pad_D
from target_patterns import center_cross, everyview
from target_patterns import visualize as vis_tv
import inspect
import time
import cv2
#from scipy.io import savemat
import h5py
###################
#### Config ####
target_pattern_type = "everyview"

ds = datasets.hci()#lytro()
samples = ['vinyl','museum','greek','kitchen']#ds.samples #list(set(ds.samples)-set(ds.corner_train_samples['train']))
Warping_Function = WarpingHCI_SOFT2

#############################
ref_views = ((4,4),)
crop_LF = True

pfm_dir = '/media/emre/Data/heidelberg_full_data/additional/additional_depth_disp_all_views/gt_dips/'
#pfm_dir = '/media/emre/Data/epinet_tests/25-02-11-24__lytro/pfms/'


##############################
save_output = True
vis = True
if vis:
    save_vis = save_output
  
roundMSEs = True

################
#################
curr_time = time.strftime("%d-%m-%H-%M")
curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
output_root = pfm_dir + "center_warping_" + curr_time + "/"
if save_output:
    os.mkdir(output_root)
    copyfile(curr_file,output_root + curr_file.split("/")[-1])  
    copyfile(curr_file.split("center_warping.py")[0] + "warping_lib.py", output_root + "warping_lib.py" )


for sample in samples:
    for ref_view in ref_views:
        
        disp_file = pfm_dir + sample + "_" + str(ref_view[0]) + "_" + str(ref_view[1]) + ".pfm"
        Dref = read_pfm(disp_file)
        
        if crop_LF:
            crop_size = Dref.shape
            warped_shape = Dref.shape
        else:
            crop_size = None
            warped_shape = ds.im_size
            Dref = pad_D(Dref,ds.im_size)
            
        LF = ds.readLF(sample, vis = True, crop_size = crop_size)
        LFref = LF[ref_view]
        if target_pattern_type == "cross":
            target_views = center_cross(ref_view)
        elif target_pattern_type == "everyview":
            target_views = everyview(ref_view,lf_size=ds.iar_max)
            
        vis_tv(ds.iar_max,ds.iac_max,target_views)
        

        Warpeds = np.zeros((ds.iar_max,ds.iac_max)+warped_shape+(3,))
        MSEs = np.zeros((ds.iar_max,ds.iac_max))
        PSNRs = np.zeros((ds.iar_max,ds.iac_max))
        
        if save_output:    
            output_dir = output_root + sample +"_ref_"+str(ref_view[0])+"_"+str(ref_view[1]) + "/"
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            txt_file = open(output_root + sample +"_ref_"+str(ref_view[0])+"_"+str(ref_view[1]) + "_info.txt","w") 
        for target_view in target_views:
            
            start = time.time()
#            if Warping_Function == Warping_HCI_SOFT2_CRIT:
#                Warped = Warping_Function(LFref,Dref,target_view,ref_view,LFtarget=LF[target_view])
#            Warping_Function in [WarpingHCI_SOFT2, WarpingHCI_SOFT3]:
            Warped = Warping_Function(LFref,Dref,target_view,ref_view)#,lower_disp_ratio=lower_disp_ratio,max_num_disps=max_num_disps)
            end = time.time()
            time_info ="warping took " + str(np.around(end-start,decimals=2)) + " seconds"
            print(time_info)
            print(target_view)
            MSE = computeMSE(Warped,LF[target_view])            
            Warpeds[target_view] = Warped
            Imax = np.max(LF[target_view])
            PSNR = 10*np.log10(Imax**2/MSE)
            print ("MSE: " + str(MSE) + "  PSNR: " + str(PSNR))
            if vis:
                plt.imshow(Warped/255)
                plt.show()
                if save_vis and save_output:
                    plt.imsave(output_dir + sample + "_" + str(target_view[0]) + "_" + str(target_view[1]) + "_PSNR_" + str(np.around(PSNR,3)) + ".png",
                               Warped/255)
                    
            MSEs[target_view] = np.around(MSE, decimals = 3)
            PSNRs[target_view] = PSNR
            if save_output:
                txt_file.write("target_view: " + str(target_view) + " PSNR: " + str(np.around(PSNR,decimals=3))+"\n")
                txt_file.write(time_info + "\n")
        if save_output:
            txt_file.close()
    if save_output:        
        np.save(output_dir + "PSNRs.npy",PSNRs)







