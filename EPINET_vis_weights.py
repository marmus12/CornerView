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
import os , sys
import cv2
import h5py
import time
from scipy.io import savemat
from epinet_fun.func_pfm import write_pfm
#from epinet_fun.func_makeinput import make_epiinput
from epinet_fun.func_makeinput import make_multiinput

from epinet_fun.func_vis_epinetmodel import define_epinet
from tensorflow import keras
#from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from shutil import copyfile
import inspect

sys.path.append('/home/emre/Documents/kodlar/')
from vis_and_understand import visualize

###CONFIG####
vis_featmaps = True
vis_and_understand = False
#ds = "lytro_lenslet"
#sample_names = ['I09_Fountain_Vincent_2','I01_Bikes','I02_Danger_de_Mort','I04_Stone_Pillars_Outside']
#sample_names = ['I01_Bikes']
ds= 'hci' #'hci'  # lytro, hci, lytro_lenslet 
sample_names = ['greek'] #['bedroom','bicycle','herbs','origami']

save_output = True



path_weight = '/media/emre/Data/from_epinet/epinet_checkpoints/pretrained_9x9.hdf5'


#center_offsets = ((0,0),(2,-2),(-2,2),(2,2),(-2,-2))

'''set Model parameters )'''

model_conv_depth=7
model_filt_num=70
model_cw = 9
############




if save_output:
    test_dir = "/media/emre/Data/epinet_tests/vis/"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    output_dir = test_dir + '__' + time.strftime("%d-%m-%H-%M")+'__'+ds+'/'


if ds=='lytro_lenslet':
    ds_root =  "/media/emre/Data/Lytro_lenslet/"
    
    dir_LFimages = []
    for name in sample_names:
        dir_LFimages.append(ds_root + name)          
    (image_h,image_w)=cv2.imread(dir_LFimages[0]+'/000_000.ppm').shape[0:2]
    
elif ds=='hci':
    ds_root = "/media/emre/Data/heidelberg_full_data/additional/"
    
    dir_LFimages = []
    for name in sample_names:
        dir_LFimages.append(ds_root + name)          
    (image_h,image_w)=cv2.imread(dir_LFimages[0]+'/input_Cam000.png').shape[0:2]


Setting02_AngualrViews = range(model_cw)  # number of views ( 0~8 for 9x9 ) 


#if __name__ == '__main__':
    
# Input : input_Cam000-080.png
# Depth output : image_name.pfm
if save_output:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
#    if not os.path.exists(output_dir+'pfms/'):
#        os.makedirs(output_dir+'pfms/')   
#    
    curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
    copyfile(curr_file,output_dir + curr_file.split("/")[-1])    
    
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
#    dir_LFimages=['/media/emre/Data/heidelberg_full_data/additional/greek']
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


img_scale=1 #   1 for small_baseline(default) <3.5px, 
            # 0.5 for large_baseline images   <  7px
            
img_scale_inv=int(1/img_scale)



model_learning_rate=0

model,feat_names=define_epinet(round(img_scale*image_h),
                        round(img_scale*image_w),
                        Setting02_AngualrViews,
                        model_conv_depth, 
                        model_filt_num,
                        model_learning_rate)



''' Model Initialization '''

model.load_weights(path_weight)
#dum_sz=model.input_shape[0]
#dum=np.zeros((1,dum_sz[1],dum_sz[2],dum_sz[3]),dtype=np.float32)
#dummy=model.predict([dum, dum, dum],batch_size=1) 


f1 = h5py.File(path_weight,'r+')
"""  Depth Estimation  """




if ds == "hci":
#    corner_dict = {(0,0):"NW", (0,8):"NE", (8,0):"SW", (8,8):"SE" }
#    assert corner_code == corner_dict[corner_coords]
    center_offset = (0,0)
    
elif ds == "lytro_lenslet":
    an = 15
#    corner_x, corner_y = corner_coords
#    center_x = corner_x + (model_cw-1)//2
#    center_y = corner_y + (model_cw-1)//2            
    center_offset = (0,0)
#    center_os_x = center_x - (an - 1)//2
#    center_os_y = center_y - (an - 1)//2    
#    center_offset = (center_os_x, center_os_y)
    
#if subsample_type == "center":


print ( "center offset: " )
print ( center_offset )
for image_path in dir_LFimages:


    (val_90d , val_0d, val_45d, val_M45d, impaths)=make_multiinput(image_path,
                                                         image_h,
                                                         image_w,
                                                         Setting02_AngualrViews,
                                                         ds=ds,center_offsets=center_offset, net_type="original", corner_code = None)

    start=time.clock() 
    
    all_outputs = model.predict([ val_90d[:,::img_scale_inv,::img_scale_inv], 
                                        val_0d[:,::img_scale_inv,::img_scale_inv], 
                                        val_45d[:,::img_scale_inv,::img_scale_inv], 
                                        val_M45d[:,::img_scale_inv,::img_scale_inv]], 
                                        batch_size=1)
    # predict
    val_output_tmp=all_outputs[-1] 
                                      
    runtime=time.clock() - start
    plt.imshow(val_output_tmp[0,:,:,0])
    if save_output:
        plt.savefig(output_dir + image_path.split('/')[-1] + '.png')
    plt.show()
    print("runtime: %.5f(s)" % runtime)
     



if vis_featmaps:
    plt_vis = True
    matsave = True
    num_filts = 70
    for feat_stage in range(4):
        for i_filt in range(num_filts): 
            feat_name = feat_names[feat_stage]
            save_name = output_dir + feat_name + "_" +str(i_filt)            
            if plt_vis:
                fig,ax = plt.subplots(figsize=(10,10))
                ax.imshow(all_outputs[feat_stage][0,:,:,i_filt])  
                plt.imsave(save_name+".png",all_outputs[feat_stage][0,:,:,i_filt]) 
                plt.close()
            if matsave:
                savemat(save_name+".mat", {"data": all_outputs[feat_stage][0,:,:,i_filt]})

    #            plt.title(feat_name)



#            plt.show()

if vis_and_understand:
    data = [ val_90d[:,::img_scale_inv,::img_scale_inv], 
             val_0d[:,::img_scale_inv,::img_scale_inv], 
             val_45d[:,::img_scale_inv,::img_scale_inv], 
             val_M45d[:,::img_scale_inv,::img_scale_inv]]
    deconv = visualize(model, data, layer_name, feature_to_visualize, visualize_mode)




















#seq_dict = {"90d":"sequential", "0d":"sequential_1",  "45d":"sequential_2" , "M45d":"sequential_3"}
##for seq_name in ["sequential","sequential_1","sequential_2"]:
#    
#for i_filt in range(num_filts):
#    for feat_name in feat_names:
#        
#        layer_name = feat_name.split("d_")[-1]
#        degree = feat_name.split("d_")[0] + "d"
#        seq_name = seq_dict[degree]
#        W = f1['model_weights'][seq_name]['4-Stream'][layer_name]['kernel:0']
#        for iar in range(9):
#            fig,ax = plt.subplots()
#            ax.imshow(W[:,:,iar,i_filt])
#            feat_name = feat_names[feat_stage]
#    #        plt.title(feat_name)
#            plt.imsave(output_dir + "W_iar_"+ str(iar)+ "_"+ degree + "_" + feat_name + "_" +str(i_filt)+".png",W[:,:,iar,i_filt])        
#            plt.close()
##            plt.show()
#



















