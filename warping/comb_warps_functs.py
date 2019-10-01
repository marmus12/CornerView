#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:09:02 2019

@author: emre
"""
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append("/home/emre/Documents/kodlar/")
#sys.path.append("/home/emre/Documents/kodlar/regions/")
#from objects import get_im_of_coords
from useful_functs import various as vrs
from glob import glob

def show_warped(Warped):
    vrs.plt_imshow(Warped*(Warped>-5)*(Warped<5))
    
def find_coords2warp_disp(knownD_coords,target_view):
    known_disp_coords = np.array(knownD_coords)
    target_view_coords = np.array(target_view)
    distances2known = np.sum((known_disp_coords - target_view_coords)**2,1)
    coords2warp_disp = tuple(known_disp_coords[np.argmin(distances2known)])
    return coords2warp_disp

def collect_warped_ims(crnr_coords,cntr_coord,corner_warp_dir,center_warp_dir,target_view,sample):
    warped_ims = dict()
    for crnr_coord in crnr_coords:
        corner_dir = glob(corner_warp_dir + 'corner_warp_'+sample+'_ref_'+str(crnr_coord[0])+'_'+str(crnr_coord[1])+'_*')[0]+'/'
        im_path = glob(corner_dir + sample+'_'+str(target_view[0])+'_'+str(target_view[1])+'_PSNR_*.png')[0]
        warped_ims[crnr_coord] = plt.imread(im_path)[:,:,0:3]
    
    center_dir = glob(center_warp_dir +sample+'_ref_'+str(cntr_coord[0])+'_'+str(cntr_coord[1]))[0]+'/'
    im_path = glob(center_dir+ sample+'_'+str(target_view[0])+'_'+str(target_view[1])+'_PSNR_*.png')[0]
    warped_ims[4,4] = plt.imread(im_path)[:,:,0:3]
    return warped_ims