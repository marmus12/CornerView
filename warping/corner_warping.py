#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:34:58 2019

@author: emre
"""

import numpy as np
from matplotlib import pyplot as plt
from warping_lib import WarpingHCI_SOFT2, computeMSE
import sys, os
sys.path.append("/home/emre/Documents/kodlar/")
sys.path.append("/home/emre/Documents/kodlar/epinet-master/epinet_fun")
from func_pfm import read_pfm
from datasets import datasets
from shutil import copyfile
from target_patterns import NW_cross, NE_cross, SW_cross, SE_cross, everyview, single_target,everyother
from target_patterns import visualize as vis_tv
import inspect
import time
import cv2
from scipy.io import savemat
import h5py
###################
#### Config ####
target_pattern_type = "everyother"#"everyview" #"single_target"
#if target_pattern_type == "single_target":
#    single_target_view = (2,1)

warping_type = 'color' #"disp" #color    

ds = datasets.hci()#datasets.hci()

samples = ['vinyl','museum','greek','kitchen']#list(set(ds.samples)-set(['I01_Bikes']))
#############################
ref_views = ((0,0),(0,8),(8,0),(8,8))
#ref_views = ((3,3),(3,11),(11,3),(11,11))

#pfm_dir = '/media/emre/Data/cepinet_runs/24-05-12-32/tests/0170__27-05-12-46__all_corners__hci/pfms/'
#pfm_dir = '/media/emre/Data/cepinet_runs/08-05-12-40_greek/tests/0034__09-05-14-02__all_corners__lytro/pfms/'
pfm_dir = '/media/emre/Data/heidelberg_full_data/additional/additional_depth_disp_all_views/gt_dips/'
##############################
LF_shape = (ds.iar_max,ds.iac_max,ds.ir_max,ds.ic_max)
save_output = True
vis = True
if vis:
    save_vis = True

roundMSEs = True

################
#################


curr_time = time.strftime("%d-%m-%H-%M")
curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
output_root = pfm_dir + "corner_warping_" + curr_time + "/"
os.mkdir(output_root)
#output_root = pfm_dir + "corner_warping_05-06-14-45/" 

copyfile(curr_file,output_root + curr_file.split("/")[-1])  
copyfile(curr_file.split("corner_warping.py")[0] + "warping_lib.py", output_root + "warping_lib.py" )

network_cam_w = 9 
corner_dict = ds.get_corner_dict(network_cam_w)


for sample in samples:
    for ref_view in ref_views:
        
        corner_code = corner_dict[ref_view]
        
        disp_file = pfm_dir + sample + "_" + str(ref_view[0]) + "_" + str(ref_view[1]) + ".pfm"
        Dref = read_pfm(disp_file)
        #Dref = cv2.resize(Dref,LF[ref_view].shape[0:2])
        LF = ds.readLF(sample, vis = True, crop_size = Dref.shape)
        
        if target_pattern_type == "cross":
            exec("target_pattern = "+ corner_code+"_cross")
            target_views = target_pattern(ref_view)
            
        elif target_pattern_type == "everyview":
            target_pattern = everyview
            target_views = target_pattern(ref_view)
            
        elif target_pattern_type == "everyother":
            target_pattern = everyother
            target_views = everyother(ref_views,lf_size=LF_shape[0])
            
        elif target_pattern_type == "single_target":
            target_views = single_target(single_target_view)
            
        
        vis_tv(ds.iar_max,ds.iac_max,target_views)
        
        
        if warping_type == "color":
            Warpeds_shape = (ds.iar_max,ds.iac_max)+Dref.shape+(3,)
        elif warping_type == "disp":
            Warpeds_shape = (ds.iar_max,ds.iac_max)+Dref.shape
            
        Warpeds = np.zeros(Warpeds_shape)
        MSEs = np.zeros((ds.iar_max,ds.iac_max))
        PSNRs = np.zeros((ds.iar_max,ds.iac_max))
        
        if save_output:    
            output_dir = output_root + "corner_warp_" + sample +"_ref_"+str(ref_view[0])+"_"+str(ref_view[1])+ "_" + time.strftime("%d-%m-%H-%M") + "/"
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            txt_file = open(output_root + sample +"_ref_"+str(ref_view[0])+"_"+str(ref_view[1]) + "_info.txt","w")                 
        for target_view in target_views:
        
#            MSE, Warpeds[target_view] = WarpingHCI(LF,Dref,target_view,ref_view)
            start = time.time()
            if warping_type == "color":
                Warped = WarpingHCI_SOFT2(LF[ref_view], Dref, target_view, ref_view)
            elif warping_type == "disp":
                Warped = WarpingHCI_SOFT2(np.expand_dims(np.copy(Dref),-1), Dref, target_view=target_view,
                                ref_view=ref_view,color_image=False,invalid_pix_val=-800)[:,:,0]
            end = time.time()
            time_info ="warping took " + str(end - start) + " seconds"
            print(time_info)
                        
            Warpeds[target_view] = Warped
            if warping_type == "color":
                MSE = computeMSE(Warped, LF[target_view])
                Imax = np.max(LF[target_view])
                PSNR = 10*np.log10(Imax**2/MSE)
                print ("MSE: " + str(MSE) + "  PSNR: " + str(PSNR))
                if vis:
                    plt.imshow(Warped/255)
                    plt.show()
                    if save_vis:
                        plt.imsave(output_dir + sample + "_" + str(target_view[0]) + "_" + str(target_view[1]) + "_PSNR_" + str(np.around(PSNR,3)) + ".png",
                                   Warped/255)
                    
                MSEs[target_view] = np.around(MSE, decimals = 3)
                PSNRs[target_view] = PSNR
                if save_output:
                    txt_file.write("target_view: " + str(target_view) + " PSNR: " + str(np.around(PSNR,decimals=3))+"\n")
                    txt_file.write(time_info + "\n")      
                    np.save(output_dir + "PSNRs.npy",PSNRs)
            elif warping_type == "disp":
                savemat(output_dir + "Warped",{"Warped":Warpeds})
        txt_file.close()    


                
        #print((MSEs*1000).astype(int))
    #    if save_output:    
    #        curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
    #        copyfile(curr_file,output_dir + curr_file.split("/")[-1])    
    #        
    #        assert save_format in ["npy","hdf5","both"]
    #        if (save_format is "npy") or (save_format is "both"):
    #            np.save(output_dir + "MSEs.npy", MSEs)
    #            np.save(output_dir + "Warpeds.npy", Warpeds)
    #        elif (save_format is "hdf5") or (save_format is "both"):
    #            hf = h5py.File(output_dir + 'MSEs_and_Warpeds.h5', 'w')
    #            hf.create_dataset('MSEs', data=MSEs)
    #            hf.create_dataset('Warpeds', data=Warpeds)
    #            hf.close()











