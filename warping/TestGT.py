#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 14:25:28 2019

@author: emre
"""

import time
import inspect
from shutil import copyfile
import os, sys
sys.path.append("/home/emre/Documents/kodlar/epinet-master/epinet_fun/")
from func_pfm import read_pfm
import numpy as np
from matplotlib import pyplot as plt
import cv2
from WarpingHCI import WarpingHCI
from ds_utils import readLF
## Configuration ###########
TEST_DATA = "hci"
sample = "town"
visLF = True

add_data_dir = '/media/emre/Data/heidelberg_full_data/additional/'
add_gt_dir = '/media/emre/Data/heidelberg_full_data/additional_depth_disp_all_views/'

pfm_dir = '/media/emre/Data/cepinet_runs/25-01-19-58/tests/0541__08-02-21-25__NW__hci/'

disparity_file = pfm_dir + "pfms/town.pfm"

# reference view:
iar0 = 0
iac0 = 0 

##############################

def define_test_views(iar_max,iac_max): 
    TA=np.zeros((iar_max,iac_max))
    TA[iar0:(iar0+iar_max),iac0] = 1
    TA[iar0,iac0:(iac0+iac_max)] = 1
    for ii in DISPL_RANGE:
        TA[iar0+ii,iac0+ii] = 1
    
    return TA

curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
curr_date = time.strftime("%d-%m-%H-%M")  

output_dir = pfm_dir+"warping_test_"+curr_date+"/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
infofile = output_dir + 'info.txt'

curr_file_name = curr_file.split('/')[-1]
copyfile(curr_file, output_dir + curr_file_name)

if TEST_DATA == "hci":
    DIR_TEST = add_data_dir
    iar_max = 9
    iac_max = 9
    DISPL_RANGE = range(iar_max)
    nr = 512
    nc = 512
    
###READ THE LIGHT FIELD###############
LF0 = readLF(iar_max, iac_max, nr, nc, DIR_TEST, sample, vis = visLF)

###########################################


###READ THE DISPARITY ###############
D = read_pfm(disparity_file)
D = cv2.resize(D,(nr, nc))

### AUGMENT #################
#LF0prime = np.zeros_like(LF0)
#for iar = 1:iar_max
#    for icomp = 1:3
#        LF0prime(1,iar,:,:,icomp) = imrotate(squeeze(LF0(iar, 9,:,:,icomp)),90);
#    end
#end
#
#for iac = 1:iac_max
#    for icomp = 1:3
#        LF0prime(iac,1,:,:,icomp) = imrotate(squeeze(LF0(1, 10-iac,:,:,icomp)),90);
#    end
#end
###############################

### TA Boolean indicates at which positions you perform the warping
TA = define_test_views(iar_max,iac_max)
print ( "Test Array:")
print (TA)
#########################

MSEarr = np.zeros((iar_max,iac_max))
Warpeds = dict()
for iarT in range(iar_max):
    Warpeds[iarT] = dict()
    
for iarT in range(iar_max):
    for iacT in range(iac_max):
        if TA[iarT,iacT] == 1 :
#            [MSE1, Warped] = WarpingHCI(LF0prime, np.rot90(D), nr,nc, iarT, iacT, iar0,iac0 )
            [MSE1, Warped] = WarpingHCI(LF0, D, nr, nc, iarT, iacT, iar0,iac0 )
            MSEarr[iarT,iacT] = MSE1
            Warpeds[iarT][iacT] = Warped

print(MSEarr) 
np.savetxt( output_dir + "MSEarr.txt", MSEarr, fmt = '%.3f')




