# -*- coding: utf-8 -*-
"""

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

"""

#import numpy as np
import numpy as np
import os, sys
from datasets import hci, lytro
import cv2

import time
from epinet_fun.func_pfm import write_pfm
#from epinet_fun.func_makeinput import make_epiinput
from epinet_fun.func_makeinput import make_multiinput
from epinet_usefuls import infer_cos_from_corner
from epinet_fun.func_cepinetmodel import define_cepinet
from tensorflow import keras
#from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from shutil import copyfile
import inspect
from glob import glob



###CONFIG####
os.environ["CUDA_VISIBLE_DEVICES"]="0"
ds= hci("/path/to/heidelberg_full_data/") #'hci'  # lytro, hci, lytro_lenslet 

samples = ds.samples
#samples = ["town","kitchen","museum","vinyl"]
run_dir = '/media/emre/Data/cepinet_runs/08-05-12-40_greek/'
model_iter = '0034'

corner_coordss = ((11,11),)#"all"
#corner_coords = (0,0) # for hci: 0,0 : NW   0,8: NE   8,0: SW  8,8: SE
#corner_code = "SE"

'''set Model parameters '''

model_conv_depth=7
model_filt_num=70
model_cw = 9

############


path_weight = glob(run_dir + 'logs/*'+model_iter+"*")[0]




test_dir = run_dir + "tests/"
output_dir = test_dir + model_iter + '__' + time.strftime("%d-%m-%H-%M")+'__all_corners__'+ds.ds_name+'/'
#output_dir = '/media/emre/Data/cepinet_runs/08-05-12-40_greek/tests/0034__09-05-14-02__all_corners__lytro/'


    
Setting02_AngualrViews = range(model_cw)  # number of views ( 0~8 for 9x9 ) 

#if __name__ == '__main__':
    
# Input : input_Cam000-080.png
# Depth output : image_name.pfm
if not os.path.exists(output_dir):
    os.makedirs(output_dir)   
if not os.path.exists(output_dir+'pfms/'):
    os.makedirs(output_dir+'pfms/')   
    
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

        
        
        
'''
/// Setting 2. Angular Views 



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


img_scale=1 #   1 for small_baseline(default) <3.5px, 
            # 0.5 for large_baseline images   <  7px
            
img_scale_inv=int(1/img_scale)



model_learning_rate=0


corner_dict = ds.get_corner_dict(model_cw)


if corner_coordss == "all":
    corner_coordss = corner_dict.keys() 


for corner_coords in corner_coordss:

    corner_code = corner_dict[corner_coords] 
    image_h = ds.ir_max
    image_w = ds.ic_max
    if corner_code in ("NE","SW"):
        model_512=define_cepinet(round(img_scale*image_w),
                                round(img_scale*image_h),
                                model_cw,
                                model_conv_depth, 
                                model_filt_num,
                                model_learning_rate)
    else:
        model_512=define_cepinet(round(img_scale*image_h),
                                round(img_scale*image_w),
                                model_cw,
                                model_conv_depth, 
                                model_filt_num,
                                model_learning_rate)    
    
    
    ''' Model Initialization '''
    
    model_512.load_weights(path_weight)
    dum_sz=model_512.input_shape[0]
    dum=np.zeros((1,dum_sz[1],dum_sz[2],dum_sz[3]),dtype=np.float32)
    dummy=model_512.predict([dum, dum, dum],batch_size=1) 
    
    
    
    """  Depth Estimation  """
 
    center_offset = infer_cos_from_corner(corner_coords,corner_code,an=ds.iar_max,model_cw=model_cw)
        
    
    print ( "center offset: " )
    print ( center_offset )
    print ("corner code:")
    print(corner_code)
    for sample in samples:
    
    
        (val_90d , val_0d, val_M45d, impaths)=make_multiinput(sample,
                                                             Setting02_AngualrViews,
                                                             ds=ds,center_offsets=center_offset, net_type="corner", corner_code = corner_code)
    
        start=time.clock() 
        
        # predict
        val_output_tmp=model_512.predict([ val_90d[:,::img_scale_inv,::img_scale_inv], 
                                            val_0d[:,::img_scale_inv,::img_scale_inv], 
                                          val_M45d[:,::img_scale_inv,::img_scale_inv]], 
                                          batch_size=1); 
        
        
                                          
        if corner_code=="NE":        
            val_output_tmp = np.rot90(val_output_tmp,axes=(2,1))
    
    
        elif corner_code=="SW":        
            val_output_tmp = np.rot90(val_output_tmp,axes=(1,2))
    
    
        elif corner_code=="SE":        
            val_output_tmp = np.rot90(val_output_tmp,k=2,axes=(2,1))
                                                  
                                          
                                          
        runtime=time.clock() - start
        plt.imshow(val_output_tmp[0,:,:,0])
        
        plt.imsave(output_dir + sample + "_" + str(corner_coords[0])+ "_" + str(corner_coords[1]) + '.png',val_output_tmp[0,:,:,0])
        plt.show()
        print("runtime: %.5f(s)" % runtime)
         
        # save .pfm file
        pfm_path = output_dir+"pfms/"+ sample + "_" +str(corner_coords[0])+ "_" + str(corner_coords[1]) + '.pfm'
        write_pfm(val_output_tmp[0,:,:,0], pfm_path )
        print('pfm file saved in  ' + pfm_path)
    

