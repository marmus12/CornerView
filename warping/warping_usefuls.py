#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:57:43 2019

@author: emre
"""

import numpy as np

def extract_PSNRs(corner_impaths, LF_shape=(9,9)):
    PSNRs = np.zeros(LF_shape,float)
    for corner_impath in corner_impaths:
        infolist = corner_impath.split('/')[-1].split('.png')[0].split('_')
        target_view = (int(infolist[1]),int(infolist[2]))
        PSNR = float(infolist[-1])
        PSNRs[target_view] = PSNR
    return PSNRs

def find_occluded_pixels(gtWarpeds):
    occluded_pixels = dict()
    for target_view in gtWarpeds.keys():
        occluded_pixels[target_view] = np.where(gtWarpeds[target_view]==-1)
    return occluded_pixels


def pad_D(Dref,target_shape):
    
    tw,th = target_shape
    Dw, Dh = Dref.shape
    
    w_delta = (tw-Dw)//2
    h_delta = (th-Dh)//2
    padded_D = np.zeros(target_shape)
        
    padded_D[w_delta:w_delta+Dw,h_delta:h_delta+Dh] = Dref
    
    padded_D[0:w_delta,h_delta:h_delta+Dh] = Dref[0,:]
    padded_D[w_delta+Dw:,h_delta:h_delta+Dh] = Dref[-1,:]

    padded_D[:,0:h_delta] = padded_D[:,h_delta:h_delta+1]
    padded_D[:,h_delta+Dh:] = padded_D[:,h_delta+Dh-1:h_delta+Dh]    

#    plt.imshow(padded_D)
#    plt.show()
    return padded_D

