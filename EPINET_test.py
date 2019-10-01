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
sys.path.append("/home/emre/Documents/kodlar/datasets/")
from datasets import hci, lytro, hdca_set2, leiria2
import cv2

import time
from epinet_fun.func_pfm import write_pfm
#from epinet_fun.func_makeinput import make_epiinput
from epinet_fun.func_makeinput import make_multiinput
#from epinet_fun.func_epinetmodel import layer1_multistream
#from epinet_fun.func_epinetmodel import layer2_merged
#from epinet_fun.func_epinetmodel import layer3_last
from epinet_fun.func_epinetmodel import define_epinet
from tensorflow import keras
from epinet_usefuls import infer_cos_from_cc

import matplotlib.pyplot as plt
from glob import glob

from shutil import copyfile 
import inspect

###CONFIG#########################################################

ds = lytro()#hci()

crop_size = (ds.im_size[0]+22,ds.im_size[1]+22) #None#None #(int(ds.ir_max//3.7),int(ds.ic_max//3.7))#(1024,1024)
crop_or_resize = "pad"#None # "resize"
sample_names = ds.samples #list(set(ds.samples)-set(ds.corner_train_samples['train']))# ['greek'] #["img1","img2","img3"]

path_weight = '/media/emre/Data/from_epinet/epinet_checkpoints/pretrained_9x9.hdf5'
#run_dir = '/media/emre/Data/epinet_runs/09-05-13-05_greek/'
#epoch = ''


os.environ["CUDA_VISIBLE_DEVICES"]="0"
############

#test_dir = run_dir + 'tests/'
test_dir = '/media/emre/Data/epinet_tests/'
#output_dir = test_dir + epoch + '__' + time.strftime("%d-%m-%H-%M")+'__'+ds.ds_name+'/'
output_dir = test_dir + 'ori__' + time.strftime("%d-%m-%H-%M")+'__'+ds.ds_name+'/'
pfm_dir = output_dir + "pfms/"

model_cw = 9


#center_coords = [(ds.iar_max//2,ds.iac_max//2)] 

center_coords = []
for iar_center in range(model_cw//2,(ds.iar_max-model_cw//2)):
    for iac_center in range(model_cw//2,(ds.iac_max-model_cw//2)):
        center_coords.append((iar_center,iac_center))
##################################################################

#path_weight = glob(run_dir + 'logs/*'+ epoch +'*.hdf5')[0]

    
# Input : input_Cam000-080.png
# Depth output : image_name.pfm
if not os.path.exists(output_dir):
    os.makedirs(output_dir)   
if not os.path.exists(pfm_dir):
    os.makedirs(pfm_dir)       
    
curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
copyfile(curr_file,output_dir + curr_file.split("/")[-1])       
# GPU setting ( gtx 1080ti - gpu0 ) 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    



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
Setting02_AngualrViews = range(model_cw)  # number of views ( 0~8 for 9x9 ) 




img_scale=1 #   1 for small_baseline(default) <3.5px, 
            # 0.5 for large_baseline images   <  7px
            
img_scale_inv=int(1/img_scale)



''' Define Model ( set parameters )'''

model_conv_depth=7
model_filt_num=70
model_learning_rate=0.1**5

if crop_size is None:
    input_ir, input_ic = ds.ir_max, ds.ic_max
else:
    input_ir, input_ic = crop_size
    
model_512=define_epinet(round(img_scale*input_ir),
                        round(img_scale*input_ic),
                        Setting02_AngualrViews,
                        model_conv_depth, 
                        model_filt_num,
                        model_learning_rate)



''' Model Initialization '''

model_512.load_weights(path_weight)
dum_sz=model_512.input_shape[0]
dum=np.zeros((1,dum_sz[1],dum_sz[2],dum_sz[3]),dtype=np.float32)
dummy=model_512.predict([dum,dum, dum,dum],batch_size=1) 

''' Tensorboard '''
#
#tbCallback = keras.callbacks.TensorBoard(log_dir='D:\deneme', histogram_freq=0,  
#      write_graph=True, write_images=True)    
#
#tbCallback.set_model(model_512)
"""  Depth Estimation  """

for center_coord in center_coords:
    center_offset = infer_cos_from_cc(center_coord,ds)
#for center_offset in center_offsets:
    subsample = ("center",center_offset) #"center" #"upperleft"
    print ( "center offset: " )
    print ( center_offset )
#    for image_path in dir_LFimages:
    for sample in sample_names:    
    
        (val_90d , val_0d, val_45d, val_M45d,impaths)=make_multiinput(sample,
                                                             Setting02_AngualrViews,
                                                             ds=ds, center_offsets = center_offset, net_type='original', 
                                                             crop_size=crop_size, crop_or_resize=crop_or_resize)
    
        start=time.clock() 
        
        # predict
        val_output_tmp=model_512.predict([ val_90d[:,::img_scale_inv,::img_scale_inv], 
                                            val_0d[:,::img_scale_inv,::img_scale_inv], 
                                           val_45d[:,::img_scale_inv,::img_scale_inv], 
                                          val_M45d[:,::img_scale_inv,::img_scale_inv]], 
                                          batch_size=1); 
                                          
        runtime=time.clock() - start
        plt.imshow(val_output_tmp[0,:,:,0])
        
        iar_center, iac_center = center_coord
        write_name = sample +"_"+ str(iar_center) + "_" + str(iac_center)
        
        plt.imsave(output_dir + write_name +'.png',val_output_tmp[0,:,:,0])
        plt.show()
        print("runtime: %.5f(s)" % runtime)
         
        # save .pfm file
        pfm_path = pfm_dir + write_name + '.pfm'
        write_pfm(val_output_tmp[0,:,:,0], pfm_path)
        print('pfm file saved in  ' + pfm_path)
    

