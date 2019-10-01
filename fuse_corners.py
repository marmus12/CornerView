#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:31:16 2019

@author: emre
"""

import numpy as np
from glob import glob
from epinet_fun.func_pfm import read_pfm, write_pfm
import sys
sys.path.append('/home/emre/Documents/kodlar/')
from warping import warping_lib as warp
from useful_functs.various import plt_imshow, mkdir
import time, inspect
from shutil import copyfile
##CONFIG####
corner_res_dir = '/media/emre/Data/cepinet_runs/08-05-12-40/tests/0034__09-05-11-44__all_corners__hci/'
center_res_dir = '/media/emre/Data/epinet_runs/09-05-13-05/tests/10-05-11-28__hci/'
sample = 'greek'
center_view = (4,4)

inv_pix_val = -800

save_output = True

###########
if save_output:
    curr_time = time.strftime("%d-%m-%H-%M")
    output_dir = corner_res_dir + "fuse_corners_"+curr_time+"/"
    mkdir(output_dir)
    curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
    copyfile(curr_file,output_dir + curr_file.split("/")[-1])    
        
def show_warped(Warped):
    plt_imshow(Warped*(Warped>-5)*(Warped<5))
    
def construct_coord2im_dict(pfms_dir):
    pfm_files = glob(pfms_dir+'*.pfm')
    coords = []
    for pfm_file in pfm_files:    
        nums = pfm_file.split('.pfm')[0].split('_')[-2:]
        coords.append((int(nums[0]),int(nums[1])))
    
    coord2path=dict()
    for k,coord in enumerate(coords):
        coord2path[coord]=read_pfm(pfm_files[k])
        
    return coord2path


pfms_dir = corner_res_dir + 'pfms/'
coord2im = construct_coord2im_dict(pfms_dir)

## warp the corners:
Warpeds=dict()
for corner_coord in coord2im.keys(): #[(0,0)]:
    print("doing.." + str(corner_coord))
    D = coord2im[corner_coord]
    LFref = np.expand_dims(np.copy(D),-1)
    Warped = warp.WarpingHCI_SOFT2(LFref, D, target_view=center_view,
                                ref_view=corner_coord,color_image=False,invalid_pix_val=inv_pix_val)

    show_warped(Warped[:,:,0])
    Warpeds[corner_coord] = Warped

### read the center:
    
center_pfms_dir = center_res_dir + 'pfms/'


print("doing.." + str(center_view))
center_D = read_pfm(glob(center_pfms_dir+'*.pfm')[0])
Warpeds[center_view] = np.expand_dims(center_D,-1)


#### fuse:
Warped_ims=np.concatenate(list(Warpeds.values()),-1)
#fused_Warped = np.median(Warped_ims*(Warped_ims!=inv_pix_val),-1)
fused_Warped = np.zeros_like(center_D)
for ir in range(D.shape[0]):
    for ic in range(D.shape[1]):
        vals = Warped_ims[ir,ic,:]
        valid_vals = vals[vals!=inv_pix_val]
        fused_Warped[ir,ic]=np.median(valid_vals)

## show end result
plt_imshow(center_D)
plt_imshow(fused_Warped)

if save_output:
    write_pfm(fused_Warped,output_dir + sample + '_'+str(center_view[0])+'_'+str(center_view[1])+'.pfm') 












    
    
    