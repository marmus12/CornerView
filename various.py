#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:13:23 2019

@author: emre
"""

def sort_list_by_order(llist,order):
    new_list=[]
    for ind in order:
        new_list.append(llist[ind])
    return new_list

def exclude_elt_from_list(in_list,elt):
    return [idd for idd in in_list if idd != elt]     

def exclude_elts_from_list(in_list,elts):

    for elt in elts:
        new_list = [idd for idd in in_list if idd != elt]     
        in_list = np.copy(new_list)
        
    return in_list

import os
import pickle
import random 
from matplotlib import pyplot as plt

def plt_imshow(im,figsize=(7,7),horizontal=True,cmap=None):
#    fig,ax = plt.subplots(figsize=figsize);ax.imshow(im);plt.show()     
    if type(im) != list:
        fig,ax = plt.subplots(figsize=figsize);ax.imshow(im,cmap=cmap);plt.show()   

    else:
        if horizontal:
            ncols = len(im)
            nrows = 1
        else:
            ncols = 1
            nrows = len(im)
            
        fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)       
        for i in range(nrows*ncols):
            axs[i].imshow(im[i],cmap=cmap)
        plt.show()

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def log_info(strng,info_file):       
    print(strng)
    info_file.write(strng + "\n")
##########################################


def print_dict(adict):
    
    for key in adict.keys():
        print( str(key) + ": " + str(adict[key]) + "\n")
    
def center_crop2d(im,crop_size):
    crw,crh = crop_size
    imw, imh = im.shape[0:2]     
    d_w, d_h = (imw - crw)//2 , (imh - crh)//2
    
    if len(im.shape)==2:
        im2 = im[d_w:(d_w + crw), d_h:(d_h + crh)]    
    else:
        im2 = im[d_w:(d_w + crw), d_h:(d_h + crh),:]           
    return im2

def upper_crop2d(im,crop_size):
    crw,crh = crop_size
    imw, imh = im.shape[0:2]     
    d_h =  (imh - crh)//2
    
    if len(im.shape)==2:
        im2 = im[0:(crw), d_h:(d_h + crh)]    
    else:
        im2 = im[0:(crw), d_h:(d_h + crh),:]           
    return im2




def random_color(seed=None):

    levels = range(32,256,16)
    color = tuple()
    for r in range(3):
        if seed!=None:
            random.seed(a=seed+r)        
        color = color+(random.choice(levels),)
    return color

import numpy as np



def pad_sides2d(im,crop_size):
    crw,crh = crop_size
    dtype = im.dtype
    imw, imh = im.shape[0:2]   
    if len(im.shape)>2:
        nch = im.shape[2]
    else:
        nch = 1
    
    nim = np.zeros(crop_size+(nch,),dtype=dtype)
    d_w, d_h = (crw-imw)//2 , (crh-imh)//2
    nim[d_w:(d_w+imw),d_h:(d_h+imh),:]=im
    return nim
    
    
    
def np_summarize_array(arr):
    print("mean: " +  str(np.mean(arr)))
    print("max: " +  str(np.max(arr)))
    print("min: " +  str(np.min(arr)))
    print("median: " +  str(np.median(arr)))
    
def np_find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def paint(D,seed=None,show_0_and_below=False):
    
    qlevels = np.unique(D)
    n_qlev = len(qlevels)
    colors = np.zeros((n_qlev,3),int)
    
    for iql in range(n_qlev):
        if seed==None:
            colors[iql,:] = random_color()
        else:
            colors[iql,:] = random_color(seed+iql)            
        
    painteD = np.zeros(D.shape + (3,),int)     
    if show_0_and_below:    
        for iql,ql in enumerate(qlevels):   
                painteD[np.where(D==ql)]=colors[iql,:]        
    else:            
        for iql,ql in enumerate(qlevels): 
            if ql>0:          
                painteD[np.where(D==ql)]=colors[iql,:]
            
    return painteD

def paint2(D,seed=None,show_0_and_below=False):
    
    qlevels = np.unique(D)
    max_qlev = np.max(qlevels)
    colors = np.zeros((max_qlev+1,3),int)
    
    for iql in range(max_qlev):
        if seed==None:
            colors[iql,:] = random_color()
        else:
            colors[iql,:] = random_color(seed+iql)            
        
    painteD = np.zeros(D.shape + (3,),int)     
    if show_0_and_below:    
        for ql in qlevels:   
                painteD[np.where(D==ql)]=colors[ql,:]        
    else:            
        for ql in qlevels: 
            if ql>0:          
                painteD[np.where(D==ql)]=colors[ql,:]
            
    return painteD

def bool_mask_from_inds(inds,mask_size):
    mask = np.zeros(mask_size,bool)
    mask[inds] = True
    return mask




def primes(n):
    # Initialize a list
    primes = []
    count = 0
    possiblePrime = 2
    while count<n:        
        # Assume number is prime until shown it is not. 
        isPrime = True
        for num in range(2, possiblePrime):
            if possiblePrime % num == 0:
                isPrime = False
          
        if isPrime:
            primes.append(possiblePrime)
            count += 1
        possiblePrime += 1
        
    return primes



if __name__ == "__main__":
    
    #EXAMPLE 
    im1 =  np.arange(1200).reshape((20,20,3))
    plt_imshow(im1)
    
    im2 = pad_sides2d(im1,(30,30))

    plt_imshow(im2)





