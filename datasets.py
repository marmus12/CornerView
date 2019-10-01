#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:16:44 2019

@author: emre
"""
from glob import glob
import os, sys
from matplotlib import pyplot as plt
import imageio
import numpy as np
sys.path.append("/home/emre/Documents/kodlar/epinet-master/epinet_fun")
from func_pfm import read_pfm

class lytro():
    
    def __init__(self,ds_dir='/media/emre/Data/Lytro_lenslet/'):
        self.ds_name = "lytro"        
        self.ds_dir = ds_dir        
        self.samples = os.listdir(ds_dir)
        
        self.iar_max = 15
        self.iac_max = 15
        
        self.ic_max = 626
        self.ir_max = 434
        self.im_size = (self.ir_max, self.ic_max)        
        
    def get_corner_dict(self,cam_w):
#        corner_dict = np.chararray((self.iar_max,self.iac_max),itemsize=2,unicode=True)
        corner_dict = dict()
        for iar in range(self.iar_max-cam_w+1):
            for iac in range(self.iac_max-cam_w+1):
                corner_dict[iar,iac] = "NW"
                
        for iar in range(cam_w-1,self.iar_max):
            for iac in range(self.iac_max-cam_w+1):
                corner_dict[iar,iac] = "SW"           
                
        for iar in range(self.iar_max-cam_w+1):
            for iac in range(cam_w-1,self.iac_max):        
                corner_dict[iar,iac] = "NE"                
                
        for iar in range(cam_w-1,self.iar_max):
            for iac in range(cam_w-1,self.iac_max):        
                corner_dict[iar,iac] = "SE"
                
        return corner_dict

        
    def readLF(self,sample,vis,crop_size = None):        

        if crop_size is not None:    
            crw, crh = crop_size 
            LF_shape = (self.iar_max, self.iac_max, crw, crh, 3)
        else:
            LF_shape = (self.iar_max, self.iac_max, self.ir_max, self.ic_max, 3)
            
        LF = np.zeros(LF_shape)
        
        if vis:
            fig,ax = plt.subplots()
            
        for iar in range(self.iar_max):
            for iac in range(self.iac_max): 

                im_path =  self.get_im_path(sample,iar,iac)
                im = plt.imread(im_path)
                
                if crop_size is not None:
                    imw, imh = im.shape[0:2]     
                    d_w, d_h = (imw - crw)//2 , (imh - crh)//2
                    im = im[d_w:(d_w + crw), d_h:(d_h + crh),:]                    
                    
                LF[iar,iac,:,:,:] = im
                                                    
                if vis and iar == self.iar_max//2 and iac == self.iac_max//2 :
                    ax.imshow(im)
                    plt.show()
                
        return LF
    
    def get_im_path(self,sample,iar,iac):
        im_path =  self.ds_dir + sample + '/%03d_%03d.png' % (iac,iar)
        return im_path
    
class hci():
    
    def __init__(self,ds_dir="/media/emre/Data/heidelberg_full_data/",corner_val_samples = ['greek']):
        self.ds_name = "hci"
        self.ds_dir = ds_dir  
        self.add_gt_dir = "/media/emre/Data/heidelberg_full_data/additional/additional_depth_disp_all_views/"
        self.sub_dirs = glob(self.ds_dir+"*/")     
        self.samples = self.__get_samples()
        self.gt_samples = self.__get_gt_samples()
        self.corner_train_samples = self.__get_corner_train_samples(corner_val_samples)
        self.iar_max = 9
        self.iac_max = 9
        
        self.ic_max = 512
        self.ir_max = 512
        self.im_size = (self.ir_max, self.ic_max)
    
    def get_corner_dict(self,model_cw=9):
        corner_dict = {(8,0):"SW",(8,8):"SE",(0,0):"NW",(0,8):"NE"}
        return corner_dict

#    def __get_corner_train_samples(self):
#        train_samples = ['antinous','boardgames','dishes',
#                   'medieval2', 'pens', 'pillows', 'platonic', 
#                   'rosemary', 'table', 'tomb', 'tower', 'town' ]
#        val_samples = ["greek"]
#        return {"train": train_samples, "val": val_samples}

    def __get_corner_train_samples(self,val_samples=["greek"]):
        
        train_samples = ['antinous','boardgames','dishes',
                   'medieval2', 'pens', 'pillows', 'platonic', 
                   'rosemary', 'table', 'tomb', 'tower', 'town', 'greek' ]
        for val_sample in val_samples:
            train_samples.remove(val_sample)
#        train_samples = ['antinous','boardgames','dishes',
#                   'medieval2', 'pens', 'pillows', 'platonic', 
#                   'rosemary', 'table', 'tomb', 'tower', 'town' ]
#        val_samples = ["greek"]
        return {"train": train_samples, "val": val_samples}
        
    def __get_gt_samples(self):

        samples = []
        for sample in self.sample_dict.keys():
            if not('test' in self.sample_dict[sample]):
                samples.append(sample)
        return samples
    
    def __get_samples(self):
        self.sample_dict = dict()
        samples = []
        for sub_dir in self.sub_dirs:
            this_samples = os.listdir(sub_dir) 
            this_samples = self.__refine_samples(this_samples)
            for sample in this_samples:
                self.sample_dict[sample] = sub_dir
            samples = samples + this_samples

        return samples

    def __refine_samples(self,samples):
        rsamples = []
        for sample in samples:
            if ("_" not in sample) and (".txt" not in sample):
                rsamples.append(sample)
        return rsamples
    
    def load_data_for_train(self,samples):    
        num_samples = len(samples)
        traindata_all=np.zeros((num_samples, 512, 512, 9, 9, 3),np.uint8)

        
        for image_id,sample in enumerate(samples):
            print(sample)
            for i in range(self.iar_max*self.iac_max):
#                try:
                im_path = self.sample_dict[sample]+sample+ '/input_Cam0%.2d.png' % i
                tmp  = np.float32(imageio.imread(im_path)) # load LF images(9x9) 
#                except:
#                    print(self.ds_dir + dir_LFimage+'/input_Cam0%.2d.png..does not exist' % i )
                traindata_all[image_id,:,:,i//self.iac_max,i-self.iac_max*(i//self.iac_max),:]=tmp  
                del tmp

        return traindata_all

    def load_add_depth_gts_for_train(self,samples):    
        num_samples = len(samples)
        n_views = self.iar_max*self.iac_max
        traindata_label=np.zeros((num_samples, self.ir_max, self.ic_max, n_views),np.float32)
        

        for image_id,sample in enumerate(samples):

            print("loading additional gt.. " + sample)
            for i in range(n_views):

                tmp  = np.float32(read_pfm(self.add_gt_dir +sample+'/gt_disp_lowres_Cam0%.2d.pfm' %i)) # load LF disparity map
       
                traindata_label[image_id,:,:,i]=tmp  
                del tmp

        return traindata_label    

    def load_center_depth_gts(self,samples):    
        num_samples = len(samples)
        traindata_label=np.zeros((num_samples, self.ir_max, self.ic_max),np.float32)
        
        for image_id,sample in enumerate(samples):

            print("loading center view gt.. " + sample)
            disp_path = self.sample_dict[sample] + sample + "/gt_disp_lowres.pfm"   
            traindata_label[image_id,:,:] = np.float32(read_pfm(disp_path))

        return traindata_label   
    
    def readLF(self,sample,vis,crop_size = None):        

        if crop_size is not None:    
            crw, crh = crop_size 
            LF_shape = (self.iar_max, self.iac_max, crw, crh, 3)
        else:
            LF_shape = (self.iar_max, self.iac_max, self.ir_max, self.ic_max, 3)
            
        LF = np.zeros(LF_shape)
        
        if vis:
            fig,ax = plt.subplots()
            
        for iar in range(self.iar_max):
            for iac in range(self.iac_max): 

                im_path = self.get_im_path(sample,iar,iac)
                im = plt.imread(im_path)
                
                if crop_size is not None:
                    imw, imh = im.shape[0:2]     
                    d_w, d_h = (imw - crw)//2 , (imh - crh)//2
                    im = im[d_w:(d_w + crw), d_h:(d_h + crh),:]                    
                    
                LF[iar,iac,:,:,:] = im
                                                    
                if vis and iar == self.iar_max//2 and iac == self.iac_max//2 :
                    ax.imshow(im)
                    plt.show()
                
        return LF
        
    def get_im_path(self,sample,iar,iac):
        k = self.__convert_index(iar,iac)
        im_path = self.sample_dict[sample]+sample+'/input_Cam0%.2d.png' % k
        return im_path
    
    def get_gtD_path(self,sample,iar,iac):

        if (iar,iac) == (self.iar_max//2, self.iac_max//2):
            disp_path = self.sample_dict[sample] + sample + "/gt_disp_lowres.pfm"          
        
        else:
            k =self.__convert_index(iar,iac)
            disp_path = self.sample_dict[sample] + "additional_depth_disp_all_views/"+ sample + "/gt_disp_lowres_Cam0%.2d.pfm" % k

        return disp_path
    
    def __convert_index(self,iar,iac):
        k = self.iac_max*iar + iac 
        return k
    
    def readGTDisparity(self,sample,view):

        iar,iac = view
        disp_path = self.get_gtD_path(sample,iar,iac)
        if os.path.exists(disp_path):
            D = read_pfm(disp_path)
            return D
        else:
            return None

        
class hdca_set2():
    
    def __init__(self,ds_dir="/media/emre/Data/HDCA/set2_full/"):
        self.ds_name = "hdca_set2"
        self.ds_dir = ds_dir        
        self.samples = ["set2_full"]
        
        self.iar_max = 21
        self.iac_max = 101
        
        self.ic_max = 3976
        self.ir_max = 2652
        self.im_size = (self.ir_max, self.ic_max)

    def __crop_image(im,crop_size):
        crw,crh = crop_size
        imw, imh = im.shape[0:2]     
        d_w, d_h = (imw - crw)//2 , (imh - crh)//2
        im = im[d_w:(d_w + crw), d_h:(d_h + crh),:]       
        return im
    
    def readLF(self,vis,crop_size = None):        

        if crop_size is not None:    
            crw, crh = crop_size 
            LF_shape = (self.iar_max, self.iac_max, crw, crh, 3)
        else:
            LF_shape = (self.iar_max, self.iac_max, self.ir_max, self.ic_max, 3)
            
        LF = np.zeros(LF_shape)
        
        if vis:
            fig,ax = plt.subplots()
            
        for iar in range(self.iar_max):
            for iac in range(self.iac_max): 

                im_path = self.get_im_path(None,iar,iac)
                im = plt.imread(im_path)
                
                if crop_size is not None:
                    im = self.__crop_image(im,crop_size)
                    
                LF[iar,iac,:,:,:] = im
                                                    
                if vis and iar == self.iar_max//2 and iac == self.iac_max//2 :
                    ax.imshow(im)
                    plt.show()
                
        return LF    
    
    def get_im_path(self,sample,iar,iac):
        im_path = self.ds_dir + str(int(iac)).zfill(3)+'_'+str(int(iar)).zfill(3)+'.png'  
        return im_path        
        

class leiria2():
    
    def __init__(self,ds_dir="/media/emre/Data/Leiria_dataset2/"):
        self.ds_name = "leiria2"
        self.ds_dir = ds_dir        
        self.samples = ["img1","img2","img3"]
#        
        self.iar_max = 9
        self.iac_max = 9
#        
        self.ic_max = 1920
        self.ir_max = 1080
        self.im_size = (self.ir_max, self.ic_max)
#
    def __crop_image(im,crop_size):
#        crw,crh = crop_size
#        imw, imh = im.shape[0:2]     
#        d_w, d_h = (imw - crw)//2 , (imh - crh)//2
#        im = im[d_w:(d_w + crw), d_h:(d_h + crh),:]       
        return im
    
    def readLF(self,vis,crop_size = None):        
#
        if crop_size is not None:    
            crw, crh = crop_size 
            LF_shape = (self.iar_max, self.iac_max, crw, crh, 3)
        else:
            LF_shape = (self.iar_max, self.iac_max, self.ir_max, self.ic_max, 3)
            
        LF = np.zeros(LF_shape)
#        
        if vis:
            fig,ax = plt.subplots()
            
        for iar in range(self.iar_max):
            for iac in range(self.iac_max): 

                im_path = self.get_im_path(None,iar,iac)
                im = plt.imread(im_path)
                
                if crop_size is not None:
                    im = self.__crop_image(im,crop_size)
                    
                LF[iar,iac,:,:,:] = im
                                                    
                if vis and iar == self.iar_max//2 and iac == self.iac_max//2 :
                    ax.imshow(im)
                    plt.show()
                
        return LF    
    
    def get_im_path(self,sample,iar,iac):
        im_path = self.ds_dir + sample + '/ppm/' + str(int(iac+1)).zfill(3)+'_'+str(int(9-iar)).zfill(3)+'.png'  
        return im_path        
        
        
        
                
        
        
        
        
        
        
        
        
        
        