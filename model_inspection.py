# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:30:45 2019

@author: vhemka
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:41:04 2018

@author: shinyonsei2
"""

''' 
The order of LF image files may be different with this file.
(Top to Bottom, Left to Right, and so on..)
  
If you use different LF images, 

you should change our 'func_makeinput.py' file.

# Light field images: input_Cam000-080.png
# All viewpoints = 9x9(81)

# -- LF viewpoint ordering --
# 00 01 02 03 04 05 06 07 08
# 09 10 11 12 13 14 15 16 17
# 18 19 20 21 22 23 24 25 26
# 27 28 29 30 31 32 33 34 35
# 36 37 38 39 40 41 42 43 44
# 45 46 47 48 49 50 51 52 53
# 54 55 56 57 58 59 60 61 62
# 63 64 65 66 67 68 69 70 71
# 72 73 74 75 76 77 78 79 80
 

# We use star-shape 9x9 viewpoints 
# for depth estimation
#
# 00          04          08
#    10       13       16 
#       20    22    24 
#          30 31 32 
# 36 37 38 39 40 41 42 43 44
#          48 49 50 
#       56    58    60 
#    64       67       70 
# 72          76          80    

'''

#import numpy as np
import numpy as np
import os, sys
import cv2

import time
from epinet_fun.func_pfm import write_pfm
from epinet_fun.func_makeinput import make_epiinput
from epinet_fun.func_makeinput import make_multiinput
from epinet_fun.func_epinetmodel import layer1_multistream
from epinet_fun.func_epinetmodel import layer2_merged
from epinet_fun.func_epinetmodel import layer3_last
from epinet_fun.func_epinetmodel import define_epinet
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

#sys.path.append('C:\\Local\\vhemka\\kodlar\\useful_functs\\')
#from data_functs import read_pgm

###CONFIG####
ds='lytro_lenslet'  # lytro, hci, lytro_lenslet 
dir_output = 'D:\\deneme\\' 
sample_name = 'I02_Danger_de_Mort'


############

if ds=='lytro_lenslet':
    ds_root =  "S:\\81201_IMSIM\\Lytro_lenslet\\"
    dir_LFimages = [ds_root + sample_name]
    
    (image_h,image_w)=cv2.imread(dir_LFimages[0]+'\\000_000.ppm').shape[0:2]
    Setting02_AngualrViews = list(range(9))

else: 
    Setting02_AngualrViews = [0,1,2,3,4,5,6,7,8]  # number of views ( 0~8 for 9x9 ) 


#if __name__ == '__main__':
    
# Input : input_Cam000-080.png
# Depth output : image_name.pfm
if not os.path.exists(dir_output):
    os.makedirs(dir_output)   
    
    
# GPU setting ( gtx 1080ti - gpu0 ) 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
    



'''
/// Setting 1. LF Images Directory

Setting01_LFdir = 'synthetic': Test synthetic LF images (from 4D Light Field Benchmark)
                             "A Dataset and Evaluation Methodology for 
                             Depth Estimation on 4D Light Fields".
                            http://hci-lightfield.iwr.uni-heidelberg.de/

Setting01_LFdir = 'Lytro': Test real LF images(Lytro)

'''
#    Setting01_LFdir = 'synthetic'
#    Setting01_LFdir='Lytro'
#    
#    if(Setting01_LFdir=='synthetic'):    
#    dir_LFimages=['D:\\heidelberg_full_data\\additional\\greek']
#    image_w=512
#    image_h=512
#        
#    elif(Setting01_LFdir=='Lytro'): 
#        dir_LFimages=['lytro/2067']    
#        image_w=552
#        image_h=383  
        
        
        
        
'''
/// Setting 2. Angular Views 

Setting02_AngualrViews = [2,3,4,5,6] : 5x5 viewpoints

Setting02_AngualrViews = [0,1,2,3,4,5,6,7,8] : 9x9 viewpoints

# ------ 5x5 viewpoints -----
#                                  
#       20    22    24 
#          30 31 32 
#       38 39 40 41 42      
#          48 49 50 
#       56    58    60 
#                         
# ---------------------------                      

# ------ 9x9 viewpoints -----
# 
# 00          04          08
#    10       13       16 
#       20    22    24 
#          30 31 32 
# 36 37 38 39 40 41 42 43 44
#          48 49 50 
#       56    58    60 
#    64       67       70 
# 72          76          80       
#
# ---------------------------
'''
    
#    Setting02_AngualrViews = [2,3,4,5,6]  # number of views ( 2~6 for 5x5 )     


if(len(Setting02_AngualrViews)==5):
    path_weight='epinet_checkpoints\\pretrained_5x5.hdf5' # sample weight.    
if(len(Setting02_AngualrViews)==9 or len(Setting02_AngualrViews)==15):
    path_weight='epinet_checkpoints\\pretrained_9x9.hdf5' # sample weight.
#        path_weight='epinet_checkpoints/EPINET_train_ckp/iter0097_trainmse2.706_bp12.06.hdf5'




img_scale=1 #   1 for small_baseline(default) <3.5px, 
            # 0.5 for large_baseline images   <  7px
            
img_scale_inv=int(1/img_scale)



''' Define Model ( set parameters )'''

model_conv_depth=7
model_filt_num=70
model_learning_rate=0.1**5
model_512=define_epinet(round(img_scale*image_h),
                        round(img_scale*image_w),
                        Setting02_AngualrViews,
                        model_conv_depth, 
                        model_filt_num,
                        model_learning_rate)


num_layers = len(model_512.layers)
weights = dict()

for l,layer in enumerate(model_512.layers): 

    weights[layer.name] = layer.get_weights()


''' Model Initialization '''

model_512.load_weights(path_weight)