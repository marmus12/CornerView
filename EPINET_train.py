# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:31:58 2017

@author: shinyonsei2
"""


from __future__ import print_function


from epinet_fun.func_generate_traindata import generate_traindata_for_train
from epinet_fun.func_generate_traindata import data_augmentation_for_train
from epinet_fun.func_generate_traindata import generate_traindata512

from epinet_fun.func_epinetmodel import define_epinet
from epinet_fun.func_pfm import read_pfm
from epinet_fun.func_savedata import display_current_output
from epinet_fun.util import load_LFdata

import numpy as np
import matplotlib.pyplot as plt

import h5py
import os, sys
sys.path.append("/home/emre/Documents/kodlar/datasets")
from datasets import hci

import time
import imageio
import datetime
import threading

import inspect
from shutil import copyfile
from tensorflow import keras
import tensorflow as tf

def refine_list(loglist):
    newlist = []
    for path in loglist:
        if "iter" in path:
            newlist.append(path)
    return newlist

#####CONFIGURATION#####
#gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
#keras.backend.set_session(sess)
ds = hci(corner_val_samples=["town"])
train_samples = ds.corner_train_samples["train"]
val_samples = ds.corner_train_samples["val"]

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# change these for debug:
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
save_output = True
######
plot_losses = True

load_weight_is=False
restore_date = ""
networkname='EPINET_train'

''' 
Define Model parameters    
    first layer:  3 convolutional blocks, 
    second layer: 7 convolutional blocks, 
    last layer:   1 convolutional block
''' 
model_conv_depth=7 # 7 convolutional blocks for second layer
model_filt_num=70
model_learning_rate=0.1**4
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.1**7)
''' 
Define Patch-wise training parameters
'''        

input_size=23+2         # Input size should be greater than or equal to 23
label_size=input_size-22 # Since label_size should be greater than or equal to 1


batch_size=48     
workers_num=2  # number of threads

steps_per_epoch=10000#000 



#####################
Setting02_AngualrViews = np.array(range(ds.iar_max))  # number of views ( 0~8 for 9x9 ) 


curr_date = time.strftime("%d-%m-%H-%M")    
if load_weight_is:
    output_dir = '/media/emre/Data/epinet_runs/'+restore_date + "/"
else:
    output_dir = '/media/emre/Data/epinet_runs/'+curr_date + "/"


log_dir = output_dir+"logs/"
    
''' 
We use fit_generator to train EPINET, 
so here we defined a generator function.
'''

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def myGenerator(traindata_all,traindata_label,
                input_size,label_size,batch_size,
                Setting02_AngualrViews):  
    while 1:
        (traindata_batch_90d, traindata_batch_0d,
         traindata_batch_45d, traindata_batch_m45d, 
         traindata_label_batchNxN)= generate_traindata_for_train(traindata_all,traindata_label,
                                                                 input_size,label_size,batch_size,
                                                                 Setting02_AngualrViews)                

        (traindata_batch_90d, traindata_batch_0d,
         traindata_batch_45d,traindata_batch_m45d, 
         traindata_label_batchNxN) =  data_augmentation_for_train(traindata_batch_90d, 
                                                                  traindata_batch_0d,
                                                                  traindata_batch_45d,
                                                                  traindata_batch_m45d, 
                                                                  traindata_label_batchNxN,
                                                                  batch_size) 

        traindata_label_batchNxN=traindata_label_batchNxN[:,:,:,np.newaxis] 


        yield([traindata_batch_90d,
               traindata_batch_0d,
               traindata_batch_45d,
               traindata_batch_m45d],
               traindata_label_batchNxN)



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    


''' 
GPU setting ( Our setting: gtx 1080ti,  
                           gpu number = 0 ) 
'''






iter00=0





   


''' 
Define directory for saving checkpoint file & disparity output image
'''       
if not os.path.exists(output_dir):
    os.makedirs(output_dir)   
    

if not os.path.exists(log_dir):
    os.makedirs(log_dir)        
    
if save_output:
    curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
    copyfile(curr_file,output_dir + curr_date + "__" + curr_file.split("/")[-1])    
    
    txt_name=output_dir+'lf_%s.txt' % (networkname)            
    
    
    



#traindata_all,traindata_label=load_LFdata(dir_LFimages,hci_root)
# 
#traindata_90d,traindata_0d,traindata_45d,traindata_m45d,_ =generate_traindata512(traindata_all,traindata_label,Setting02_AngualrViews)
## (traindata_90d, 0d, 45d, m45d) to validation or test
## traindata_90d, 0d, 45d, m45d:  16x512x512x9  float32


#'''load invalid regions from training data (ex. reflective region)'''    
#boolmask_img4= imageio.imread(hci_root + 'additional/additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png')
#boolmask_img6= imageio.imread(hci_root + 'additional/additional_invalid_area/museum/input_Cam040_invalid_ver2.png')
#boolmask_img15=imageio.imread(hci_root + 'additional/additional_invalid_area/vinyl/input_Cam040_invalid_ver2.png')
#
#boolmask_img4  = 1.0*boolmask_img4[:,:,3]>0
#boolmask_img6  = 1.0*boolmask_img6[:,:,3]>0
#boolmask_img15 = 1.0*boolmask_img15[:,:,3]>0    


print('Load train data...') 

traindata_all = ds.load_data_for_train(train_samples)
train_labels = ds.load_center_depth_gts(train_samples)

train512_data = dict()
dat90, dat0, dat45, datm45, datlabel = generate_traindata512(traindata_all,train_labels,Setting02_AngualrViews)
train512_data["90d"] = dat90
train512_data["0d"] = dat0
train512_data["45d"] = dat45
train512_data["m45d"] = datm45
train512_data["label"] = datlabel

print('Load train data... Complete')  
########################
print('Load test data...') 

valdata_all = ds.load_data_for_train(val_samples)
val_labels = ds.load_center_depth_gts(val_samples)

val512_data = dict()
dat90, dat0, dat45, datm45, datlabel = generate_traindata512(valdata_all,val_labels,Setting02_AngualrViews)
val512_data["90d"] = dat90
val512_data["0d"] = dat0
val512_data["45d"] = dat45
val512_data["m45d"] = datm45
val512_data["label"] = datlabel

print('Load test data... Complete') 

  

''' 
Model for patch-wise training  
'''
model=define_epinet(input_size,input_size,
                        Setting02_AngualrViews,
                        model_conv_depth, 
                        model_filt_num,
                        model_learning_rate)
  


''' 
Model for predicting full-size LF images  
'''
image_w=512
image_h=512
model_512=define_epinet(image_w,image_h,
                        Setting02_AngualrViews,
                        model_conv_depth, 
                        model_filt_num,
                        model_learning_rate)


   
 
""" 
load latest_checkpoint
"""
if load_weight_is:
    list_name=os.listdir(log_dir)
    if(len(list_name)>=1):
        list1=refine_list(os.listdir(log_dir))
        list_i=0
        for list1_tmp in list1:
            if(list1_tmp ==  'checkpoint'):
                list1[list_i]=0
                list_i=list_i+1   
            else:
                list1[list_i]=int(list1_tmp.split('_')[0][4:])
                list_i=list_i+1            
        list1=np.array(list1) 
        iter00=list1[np.argmax(list1)]+1
        ckp_name=list_name[np.argmax(list1)].split('.hdf5')[0]+'.hdf5'
        model.load_weights(log_dir+ckp_name)
        print("Network weights will be loaded from previous checkpoints \n(%s)" % ckp_name)


my_generator = myGenerator(traindata_all,train_labels,input_size,label_size,batch_size,Setting02_AngualrViews)

best_mse=100.0
val_loss = []
train_loss = []
for iter02 in range(10000000):
    
    ''' Patch-wise training... start'''
    t0=time.time()
    
    model.fit_generator(my_generator, steps_per_epoch = steps_per_epoch, 
                        epochs = iter00+1, class_weight=None, max_queue_size=10, 
                        initial_epoch=iter00, verbose=1,workers=workers_num,callbacks=[reduce_lr])

    iter00=iter00+1
    
    
    ''' Test after N*(display_status_ratio) iteration.'''
    weight_tmp1=model.get_weights() 
    model_512.set_weights(weight_tmp1)
    train_output=model_512.predict([train512_data["90d"],train512_data["0d"],train512_data["45d"],train512_data["m45d"]],batch_size=1)
    val_output=model_512.predict([val512_data["90d"],val512_data["0d"],val512_data["45d"],val512_data["m45d"]],batch_size=1)


    ''' Save prediction image(disparity map) in 'current_output/' folder '''    
    train_error, train_bp = display_current_output(train_output, train512_data["label"], iter00, output_dir, split="train")
    val_error, val_bp = display_current_output(val_output, val512_data["label"], iter00, output_dir, split='val')


#    training_mean_squared_error_x100=100*np.average(np.square(train_error))
#    training_bad_pixel_ratio=100*np.average(train_bp)
    train_mse = np.average(np.square(train_error))
    val_mse = np.average(np.square(val_error))
    val_mean_squared_error_x100 = 100*val_mse
    val_bad_pixel_ratio=100*np.average(val_bp)
    
    val_loss.append(val_mse)
    train_loss.append(train_mse)
    if plot_losses:
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.title('train and val MSE')
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        
    if save_output:    
        save_path_file_new=(log_dir+'/epoch%04d_trainmse%.3f_bp%.2f.hdf5'  
                            % (iter00,val_mean_squared_error_x100,
                                      val_bad_pixel_ratio) )
        """ 
        Save bad pixel & mean squared error
        """        
        print(save_path_file_new)
        f1 = open(txt_name, 'a')
        f1.write('.'+save_path_file_new+'\n')
        f1.close()              
        t1=time.time()        
        
        ''' save model weights if it get better results than previous one...'''
        if(val_mean_squared_error_x100 < best_mse):
            best_mse = val_mean_squared_error_x100
            model.save(save_path_file_new)
            print("saved!!!")    
                
    
