# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 13:38:16 2018

@author: shinyonsei2
"""

import imageio, cv2
import scipy
import numpy as np
from matplotlib import pyplot as plt
import sys

sys.path.append("/home/emre/Documents/kodlar/useful_functs")
from various import center_crop2d, upper_crop2d, pad_sides2d,plt_imshow

#import os


def make_epiinput(sample, inds,view_n,RGB,corner_code, ds, crop_size=None, crop_or_resize = None): # hci
    
    if crop_size is not None:
        im_w, im_h = crop_size
    else:
        im_w, im_h = ds.ir_max, ds.ic_max
    
    cam_w = len(view_n)    
    data_tmp=np.zeros((1,im_w,im_h,cam_w),dtype=np.float32)
    i=0
    

    impaths = []    
    for indpair in inds:
        iar, iac = indpair
#        ds.get_im_path(iar,iac)

        impath = ds.get_im_path(sample,iar,iac)
 
#            
        impaths.append(impath)
        img = imageio.imread(impath)

            
        if crop_or_resize is not None:
            if crop_or_resize is "pad":
                imw,imh = img.shape[0:2]
                center_cropped = center_crop2d(img,(imw-6,imh-6))                
                cropped_im = pad_sides2d(center_cropped,crop_size)            
#                a=5
            elif crop_or_resize is "crop":
                cropped_im = center_crop2d(img,crop_size)
            elif crop_or_resize is "crop_upper":
                cropped_im = upper_crop2d(img,crop_size)                
            elif crop_or_resize is "resize":
                cropped_im = cv2.resize(img,crop_size[::-1])      
                
            tmp  = np.float32(cropped_im)            
        else:
            tmp  = np.float32(img)
        
        data_tmp[0,:,:,i]=(RGB[0]*tmp[:,:,0] + RGB[1]*tmp[:,:,1] + RGB[2]*tmp[:,:,2])/255
        i+=1
        
    
    if corner_code=="NE":        
        data_tmp = np.rot90(data_tmp,axes=(1,2))


    elif corner_code=="SW":        
        data_tmp = np.rot90(data_tmp,axes=(2,1))


    elif corner_code=="SE":        
        data_tmp = np.rot90(data_tmp,k=2,axes=(1,2))
        
        
    return data_tmp, impaths  

#def make_epiinput_lytro(image_path,seq1,image_h,image_w,view_n,RGB,corner_code):
#    data_tmp=np.zeros((1,image_h,image_w,len(view_n)),dtype=np.float32)
#    
#    i=0
#    if(len(image_path)==1):
#        image_path=image_path[0]
#    impaths = []    
#    for seq in seq1:
#        impath = image_path+'/%s_%02d_%02d.png' % (image_path.split("/")[-1],1+seq//9, 1+seq-(seq//9)*9)
#        impaths.append(impath)
#        tmp  = np.float32(imageio.imread(impath) )
#        data_tmp[0,:,:,i]=(RGB[0]*tmp[:,:,0] + RGB[1]*tmp[:,:,1] + RGB[2]*tmp[:,:,2])/255
#        i+=1
#        
#    if corner_code==None or corner_code=="NW":
#        return data_tmp, impaths 
#    
#    elif corner_code=="NE":
#        
#        a=5
#        return data_tmp, impaths  


    
def make_multiinput(image_path, view_n, ds, center_offsets, net_type, corner_code=None, vis=False, crop_size=None, crop_or_resize=None):

    nv=len(view_n)    
    an_r = ds.iar_max
    an_c = ds.iac_max
#    if ds== "lesions":
#            an = 9
#    elif ds == "lytro_lenslet":
#            an = 15
#    elif ds == "hdca":
#            an = 11
    RGB = [0.299,0.587,0.114] ## RGB to Gray // 0.299 0.587 0.114
    
    ''' data from http://hci-lightfield.iwr.uni-heidelberg.de/
    Sample images "training/dino, training/cotton" Cam000~ Cam080.png  
    We select seq of images to get epipolar images.
    For example, seq90d: Cam076.png, 67, 58, 49, 40, 31, 22, 13, 4'''
    
    # 00          04          08
    #    10       13       16 
    #       20    22    24 
    #          30 31 32 
    # 36 37 38 39 40 41 42 43 44
    #          48 49 50 
    #       56    58    60 
    #    64       67       70 
    # 72          76          80        
    osx,osy = center_offsets # center_offsets    
    d_r = (an_r - nv)//2
    d_c = (an_c - nv)//2
    
    if net_type == "original":

        seq90d = list( zip( list(range(d_r+osx,d_r+nv+osx)[::-1]), list( int((an_c-1)//2+osy) *np.ones(nv) )))  
        seq0d = list( zip( list(int((an_r-1)//2+osx)*np.ones(nv))  , list(range(d_c+osy,d_c+nv+osy)) ))
        seq45d = list( zip( list(range(d_r+osx,d_r+nv+osx)[::-1] ) , list(range( d_c+osy,d_c+nv+osy) )))
        seqM45d = list( zip( list(range(d_r+osx,d_r+nv+osx)) , list(range( d_c+osy,d_c+nv+osy) )))       

    elif net_type == "corner":

        if corner_code == "NW":                
            seq90d = list( zip( list(range(d_r+osx,d_r+nv+osx)[::-1]), list( (d_c+osy)*np.ones(nv,dtype=np.int)) ))  
            seq0d = list( zip( list((d_r+osx)*np.ones(nv,dtype=np.int))  , list(range(d_c+osy,d_c+nv+osy)) ))
            seqM45d = list( zip( list(range(d_r+osx,d_r+nv+osx)) , list(range( d_c+osy,d_c+nv+osy) )))               

        elif corner_code == "NE":
            seq90d = list( zip( list((d_r+osx)*np.ones(nv,dtype=np.int))  , list(range(d_c+osy,d_c+nv+osy)) ))
            seq0d = list( zip( list(range(d_r+osx,d_r+nv+osx)), list( (d_c+nv-1+osy)*np.ones(nv,dtype=np.int) )))  
            seqM45d = list( zip( list(range(d_r+osx,d_r+nv+osx)) , list(range( d_c+osy,d_c+nv+osy)[::-1] )))     
            
        elif corner_code == "SW":                
            seq90d = list( zip( list((d_r+nv+osx-1)*np.ones(nv,np.int))  , list(range(d_c+osy,d_c+nv+osy)[::-1]) ))
            seq0d = list( zip( list(range(d_r+osx,d_r+nv+osx)[::-1]), list( int(d_c+osy) *np.ones(nv,np.int) )))  
            seqM45d = list( zip( list(range(d_r+osx,d_r+nv+osx)[::-1]) , list(range( d_c+osy,d_c+nv+osy) )))                       

        elif corner_code == "SE":
            seq90d = list( zip( list(range(d_r+osx,d_r+nv+osx)), list( (d_c+nv+osy-1) *np.ones(nv,np.int) )))  
            seq0d = list( zip( list((d_r+nv+osx-1)*np.ones(nv,np.int))  , list(range(d_c+osy,d_c+nv+osy)[::-1]) ))
            seqM45d = list( zip( list(range(d_r+osx,d_r+nv+osx)[::-1]) , list(range( d_c+osy,d_c+nv+osy)[::-1] )))                 
                

    if net_type == "original":        
        val_90d,impaths_90d = make_epiinput(image_path,seq90d,view_n,RGB,corner_code=None, ds = ds, crop_size=crop_size, crop_or_resize=crop_or_resize)        
        val_0d,impaths_0d = make_epiinput(image_path,seq0d,view_n,RGB,corner_code=None, ds = ds, crop_size=crop_size, crop_or_resize=crop_or_resize)    
        val_45d,impaths_45d = make_epiinput(image_path,seq45d,view_n,RGB,corner_code=None, ds = ds, crop_size=crop_size, crop_or_resize=crop_or_resize)    
        val_M45d,impaths_M45d=make_epiinput(image_path,seqM45d,view_n,RGB,corner_code=None, ds = ds, crop_size=crop_size, crop_or_resize=crop_or_resize)     
        impaths = {'90d': impaths_90d, '0d': impaths_0d, '45d': impaths_45d, 'M45d': impaths_M45d}         
        
        return val_90d , val_0d, val_45d, val_M45d, impaths

    elif net_type == "corner":
        val_90d, impaths_90d=make_epiinput(image_path,seq90d,view_n,RGB,corner_code, ds = ds, crop_size=crop_size, crop_or_resize=crop_or_resize)    
        val_0d, impaths_0d=make_epiinput(image_path,seq0d,view_n,RGB,corner_code, ds = ds, crop_size=crop_size, crop_or_resize=crop_or_resize)
        val_M45d, impaths_M45d=make_epiinput(image_path,seqM45d,view_n,RGB,corner_code, ds = ds, crop_size=crop_size, crop_or_resize=crop_or_resize)  
        impaths = {'90d': impaths_90d, '0d': impaths_0d, 'M45d': impaths_M45d}    
        
        return val_90d , val_0d, val_M45d, impaths            
        

        
        
        
        
        
        
        
        
        
        
        

