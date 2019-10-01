#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:54:45 2019

@author: emre
"""
import numpy as np
from matplotlib import pyplot as plt
from shutil import copyfile
import inspect
import os
import glob
###############################################
warpings_dir = "/media/emre/Data/warpings/13-02-19-30_gamma_0.7/"
save_output = True
iar_max = 9
iac_max = 9
ndegree = 3
nr = 512
nc = 512
#################################################
def get_neigh_locs(composite,nan_loc,degree,nr,nc):
    
    neigh_array = list(range(-degree,0))+list(range(1,degree+1))
    nanr,nanc = nan_loc
    
    neigh_locs = []
    for i in neigh_array:
        for j in neigh_array:
            neigh_r = max(min(nr-1,nanr + i),0)
            neigh_c = max(min(nc-1,nanc + i),0)
            if (neigh_r,neigh_c) not in neigh_locs:
                if not np.isnan(composite[neigh_r,neigh_c,0]):
                    neigh_locs.append((neigh_r,neigh_c))
    return neigh_locs

def fill_in_the_blanks(composite):
    
    nan_locs = np.where(np.isnan(composite[:,:,0])==True)
    
    for k in range(len(nan_locs[0])):
        
        nan_loc = nan_locs[0][k],nan_locs[1][k]        
        neigh_locs = get_neigh_locs(composite,nan_loc,degree=ndegree,nr=nr,nc=nc)
        
        neigh_sum = 0
        for neigh_loc in neigh_locs:
            neigh_sum = neigh_sum + composite[neigh_loc]        
        neigh_mean = neigh_sum/len(neigh_locs)
        
        composite[nan_locs[0][k],nan_locs[1][k]] = neigh_mean
        
        
        
    return composite
    
    
if save_output:
    curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
    copyfile(curr_file,warpings_dir + curr_file.split("/")[-1])    

sample_paths = glob.glob(warpings_dir + "Warped_*")   
     
for sample_path in sample_paths:
    
    sample = sample_path.split("_")[-1].split(".npy")[0]
    
    sample_output_dir = warpings_dir + sample + "/"
    if not os.path.exists(sample_output_dir):
        os.mkdir(sample_output_dir)
        
    allWarpeds = np.load(sample_path).all()
    
    
    target_views = list(allWarpeds.keys())
        
    #fig,axs = plt.subplots(len(target_views),figsize=(55,55))
    composites = dict()
    for k,target_view in enumerate(target_views):
    
        composite = np.zeros((nr,nc,3))
        valid_locs = dict()
        valid_loc_sum = np.zeros((nr,nc))
        
        targets_ref_views = list(allWarpeds[target_view].keys())
        for ref_view in targets_ref_views:    
            valid_locs[ref_view] = np.sum(allWarpeds[target_view][ref_view],2)!=-3 
            valid_loc_sum = valid_loc_sum + valid_locs[ref_view]   
            
            this_warped = allWarpeds[target_view][ref_view]
            this_warped[np.where(this_warped==-1)] = 0
            composite = composite + this_warped
        
        composite = composite/np.tile(valid_loc_sum[:,:,np.newaxis],(1,1,3))
    #    axs[k].imshow(composites[target_view])
    
        composites[target_view] = fill_in_the_blanks(composite)
        if save_output:
            plt.imsave(sample_output_dir + "view_"+str(target_view)+".png",composites[target_view])
    #    plt.imsave()

#plt.imsave(warpings_dir + "views.png")
#plt.show()
