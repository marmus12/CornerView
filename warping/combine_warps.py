#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:49:59 2019

@author: emre
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:14:20 2019

@author: emre
"""
from skimage.segmentation import mark_boundaries, find_boundaries
import skimage.measure
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import sys
sys.path.append("/home/emre/Documents/kodlar/")
sys.path.append("/home/emre/Documents/kodlar/regions/")
from objects import get_im_of_coords
from useful_functs import various as vrs
from useful_functs.various import plt_imshow
from useful_functs.func_pfm import read_pfm
sys.path.append("/home/emre/Documents/kodlar/datasets/")
from datasets import hci
import warping_usefuls
from warping_lib import WarpingHCI_SOFT2
from comb_warps_functs import show_warped,find_coords2warp_disp,collect_warped_ims
from target_patterns import everyother
from target_patterns import visualize as vis_tv
from shutil import copyfile
import inspect
import time
from scipy.io import savemat
### CONFIGURATION ##########
    
plt_size = (5,5)
ds = hci()    
crnr_coords = ((0,0),(0,8),(8,0),(8,8))
cntr_coord = (4,4)

knownD_coords = (cntr_coord,) + crnr_coords
target_views = everyother(knownD_coords) #((1,2),)# 

samples = ['vinyl','museum','greek','kitchen']##['herbs']#list(set(ds.samples) - set(ds.corner_train_samples['train']))

#corner_pfms_dir = '/media/emre/Data/cepinet_runs/08-05-12-40_greek/tests/0034__10-05-15-19__all_corners__hci/pfms/'
#corner_warp_dir = corner_pfms_dir + 'corner_warping_14-05-15-13/'
#
#center_pfms_dir = '/media/emre/Data/epinet_runs/09-05-13-05_greek/tests/14-05-15-21__hci/pfms/'
#center_warp_dir = center_pfms_dir + 'center_warping_14-05-15-24/'
corner_pfms_dir = '/media/emre/Data/heidelberg_full_data/additional/additional_depth_disp_all_views/gt_dips/'
corner_warp_dir = corner_pfms_dir + 'corner_warping_20-06-16-34/'

center_pfms_dir = '/media/emre/Data/heidelberg_full_data/additional/additional_depth_disp_all_views/gt_dips/'
center_warp_dir = center_pfms_dir + 'center_warping_20-06-16-40/'


# DEPTH WARPING #######
Nq = 8#16
inv_pix_val = -5
##############################
curr_time = time.strftime("%d-%m-%H-%M")
curr_file = inspect.getfile(inspect.currentframe())
output_dir = corner_warp_dir + 'combine_warps_' + curr_time + '/'
vrs.mkdir(output_dir)
copyfile(curr_file,output_dir+curr_file.split('/')[-1])
num_refs = len(knownD_coords)
txt_file = open(output_dir + "info.txt","w")  

for target_view in target_views:
    for sample in samples:


        info_str = sample+'_'+str(target_view[0])+'_'+str(target_view[1])
        
        warped_ims = collect_warped_ims(crnr_coords,cntr_coord,corner_warp_dir,center_warp_dir,target_view,sample)
        keys4stacked_ws = list(warped_ims.keys())
        stacked_warpeds = np.stack(warped_ims.values())
        assert stacked_warpeds.shape[0] == num_refs
        
        D_shape = stacked_warpeds.shape[1:3]
        
        LF = ds.readLF(sample,False,D_shape)
        targetLF = LF[target_view]
        print("target view: " + str(target_view))
        vrs.plt_imshow(targetLF,plt_size)        
        Diff_Ims = np.zeros(((num_refs,)+D_shape))
        for ref_ind in range(num_refs):
            Diff_Ims[ref_ind,:,:] = np.sum(np.square(stacked_warpeds[ref_ind,:]-targetLF),-1)
            
        
        # WARP CLOSEST KNOWN DISPARITY TO TARGET VIEW####
        coords2warp = find_coords2warp_disp(knownD_coords,target_view)
        
        if coords2warp == cntr_coord:
            pfm_path = glob(center_pfms_dir+sample+'*')[0]
        else:
            pfm_path = glob(corner_pfms_dir+sample+'_'+str(coords2warp[0])+'_'+str(coords2warp[1])+'*')[0]
            
        Dref = read_pfm(pfm_path)
        targetD = WarpingHCI_SOFT2(np.expand_dims(np.copy(Dref),-1), Dref, target_view=target_view,
                                    ref_view=coords2warp,color_image=False,invalid_pix_val=inv_pix_val)[:,:,0]
        
#        plt.imsave('/home/emre/Desktop/warpedD.png',targetD)
#        assert 0
        #show_warped(targetD)
        inv_pix_mask = targetD == inv_pix_val
        #################################################
        
        ### QUANTIZE TARGET DEPTH ###############
        delta = (np.max(targetD)-np.min(targetD))/Nq
        q_targetD = np.floor((targetD-np.min(targetD))/delta)
        
#        assert np.prod((q_targetD==0 ) == inv_pix_mask) #assert that invalid pixels are labeled as 0 in quantized D

        #####CONNECTED COMPONENT ANALYSIS#######
        region_ids = np.unique(q_targetD)            
        num_total_comps = 0
        segmented_im = np.zeros(D_shape,int)
        for region_id in region_ids:
#            region_comps_im, num_region_comps = scipy.ndimage.label(q_targetD==region_id)
            region_comps_im = skimage.measure.label(q_targetD==region_id)     
            num_region_comps = len(np.unique(region_comps_im))-1
            for comp_id in range(1,num_region_comps+1):
                segmented_im[region_comps_im==comp_id] = num_total_comps + comp_id
            num_total_comps = num_total_comps + num_region_comps
            
#        plt.imsave('/home/emre/Desktop/segmented_im.png',vrs.paint(segmented_im))

        #########################################
        comp_ids = np.unique(segmented_im)
        assert list(comp_ids) == list(range(1,num_total_comps+1))
        
        best_ref_image = -500*np.ones(D_shape,int)
        for region_id in comp_ids:
            region_coords = np.where(segmented_im==region_id)
            
            region_errors = np.zeros((num_refs,))
            for ref_ind in range(num_refs):
                region_errors[ref_ind] = np.sum(Diff_Ims[ref_ind,region_coords[0],region_coords[1]])
        
            best_ref_ind = np.argmin(region_errors)
            best_ref_image[region_coords] = best_ref_ind
           
#        plt_imshow(best_ref_image,plt_size)   
        plt.imsave(output_dir+info_str+'_best_ref_im.png',best_ref_image)
        savemat(output_dir+info_str+'_best_ref_im.mat',{'bri':best_ref_image})
#        plt.imsave('/home/emre/Desktop/best_ref_im.png',vrs.paint(best_ref_image,show_0_and_below=True))
        region_reconst = -300*np.ones(targetLF.shape)
        best_refs = np.unique(best_ref_image)
        for best_ref_ind in best_refs:
            br_coords = np.where(best_ref_image==best_ref_ind)
            for ch in range(3):
                region_reconst[br_coords[0],br_coords[1],ch] = stacked_warpeds[best_ref_ind,br_coords[0],br_coords[1],ch]
        
        holes_mask = np.sum(region_reconst,-1)==0
#        hole_comps_im, num_hole_comps = scipy.ndimage.label(holes_mask)
        hole_comps_im = skimage.measure.label(holes_mask)     
        num_hole_comps = len(np.unique(hole_comps_im))-1        
        hole_ids = list(range(1,num_hole_comps+1))
        for hole_id in hole_ids: # delikler etraflarinin meaniyle doldur
            hole_mask = hole_comps_im==hole_id
            hole_bcoords = np.where(find_boundaries(hole_mask, mode = 'outer'))
            mean_pix = np.mean(region_reconst[hole_bcoords[0],hole_bcoords[1],:],0)
            region_reconst[hole_mask] = mean_pix
        #new_reg_reconst = region_reconst + np.tile(np.expand_dims(inv_pix_mask,-1),(1,1,3))*stacked_warpeds[3,:]
        MSE = np.mean(np.square(targetLF-region_reconst))
        PSNR = 10*np.log10(np.max(targetLF)**2/MSE)
        info = "region_reconst MSE : " + str(MSE) + "  PSNR: " + str(PSNR)
        print(info)
        txt_file.write(info)
        vrs.plt_imshow(region_reconst,plt_size)
        plt.imsave(output_dir+info_str+"_region_PSNR"+str(np.round(PSNR,2))+".png",region_reconst)
        
        #######MEDIAN RECONSTRUCTION OF COLOR IMAGES##############
        median_reconst = np.median(stacked_warpeds,0)
        
        non_occl_masks = np.sum(stacked_warpeds,-1)>0
        
        median_reconst = np.zeros(D_shape+(3,)) 
        for ir in range(D_shape[0]):
            for ic in range(D_shape[1]):
                non_occl=non_occl_masks[:,ir,ic]
                pix_vals = stacked_warpeds[:,ir,ic,:]
                median_pix_val = np.median(pix_vals[non_occl],0)
                median_reconst[ir,ic,:] = median_pix_val     
                
        median_reconst[np.isnan(median_reconst)] = 0
        ###################################################
        #show median stuff
        MSE = np.mean(np.square(targetLF-median_reconst))
        PSNR = 10*np.log10(np.max(targetLF)**2/MSE)
        info = "median_reconst MSE: " + str(MSE) + "  PSNR: " + str(PSNR)
        print(info)
        txt_file.write(info)
        vrs.plt_imshow(median_reconst,plt_size)
        plt.imsave(output_dir+info_str+"_median_PSNR"+str(np.round(PSNR,2))+".png",median_reconst)
        np.sum(np.abs(targetLF-median_reconst))
        
txt_file.close()
#vrs.plt_imshow(np.log(np.sum((targetLF - median_reconst)**2,-1)),plt_size)
###################################################################



#fused_Warped = np.zeros_like(warped_ims.values()[0])
#for ir in range(D.shape[0]):
#    for ic in range(D.shape[1]):
#        vals = Warped_ims[ir,ic,:]
#        valid_vals = vals[vals!=inv_pix_val]
#        fused_Warped[ir,ic]=np.median(valid_vals)


#WarpedC{1} = imread('C:\Local\tabus\Programs\2018\Emre\ELFI\boxes_ref_4_4\boxes_2_1_PSNR_30.012.png');
#
#WarpedC{2} = imread('C:\Local\tabus\Programs\2018\Emre\ELFI\corner_warp_boxes_ref_0_0_15-05-03-51\boxes_2_1_PSNR_30.563.png');
#
#WarpedC{3}= imread('C:\Local\tabus\Programs\2018\Emre\ELFI\corner_warp_boxes_ref_0_8_15-05-04-54\boxes_2_1_PSNR_28.472.png');
#
#WarpedC{4} = imread('C:\Local\tabus\Programs\2018\Emre\ELFI\corner_warp_boxes_ref_8_0_15-05-05-57\boxes_2_1_PSNR_29.866.png');
#
#WarpedC{5} = imread('C:\Local\tabus\Programs\2018\Emre\ELFI\corner_warp_boxes_ref_8_8_15-05-06-59\boxes_2_1_PSNR_28.217.png');
#
#GT1 = imread('C:\Local\tabus\Programs\2018\Emre\ELFI\input_Cam019.png');
#
#GT = GT1(12:501,12:501,:);
#
#a1 = load('C:\Users\tabus\OneDrive - TUNI.fi\EMRE\corner_center_warps\corner_warping_15-05-13-15\corner_warp_boxes_ref_0_0_15-05-13-15\Warped.mat')
#
#Disp{2} = squeeze(a1.Warped(3,2,:,:));
#
#Disp2 = Disp{2};
#
#Disp2(Disp2<-3) = -3;
#
#figure(1),clf,imagesc(Disp2),colormap(gray)
#
#% Quantize depth
#
#Nq = 32
#
#delta = (max(Disp2(:))-min(Disp2(:)))/Nq;
#
#Labels = floor((Disp2-min(Disp2(:)))/delta);
#
#figure(1),clf,imagesc( Labels),colormap(gray)
#
#figure(2),clf,imagesc( Labels),colormap(rand(10000,3))
#
#figure(3),clf,imagesc( abs(GT-WarpedC{1}) ),colormap(gray)
#
#for i = 1:5
#
#    DiffIm{i} = ( sum((WarpedC{i}-GT),3)/3 ).^2;
#
#end
#
#uL = unique(Labels(:))
#
#for ireg = 1:length(uL)
#
#    ind = find(Labels==uL(ireg));
#
#    for i = 1:5
#
#         MSEreg(ireg,i) = sum(DiffIm{i}(ind));
#
#    end
#
#     MSEreg(ireg,:)
#
#end

 