#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 14:25:22 2019

@author: emre
"""
import numpy as np
#import tensorflow as tf


#def warp(targets_ref_views, target_view, Ds, LF0):   
#    
#    
#    Warpeds = dict()
#    
#    MSEs = []
#
#    for ref_view in targets_ref_views:        
#
#        D0 = Ds[ref_view]
#        MSE1, Warpeds[ref_view] = WarpingHCI(LF0,  D0,  target_view, ref_view)
#        MSEs.append(MSE1)
#
#    return MSEs, Warpeds

def WarpingHCI(LFref, D, target_view, ref_view):
    
    LFref = (255*LFref).astype(np.uint8)
    
    nr,nc = LFref.shape[0:2]   
    Warped = -1*np.ones((nr,nc,3),dtype=int)    
    InitDisp = (-10**10)*np.ones((nr,nc))
    
    iar0, iac0 = ref_view
    
    iarT, iacT = target_view


#    InitDepth = 10**10*np.ones(nr,nc)
    for ir in range(nr):
        for ic in range(nc):
            
            vert_baseline = iarT-iar0
            horz_baseline = iacT-iac0
            ir1 = int(ir-round((D[ir,ic])*vert_baseline))
            ic1 = int(ic-round((D[ir,ic])*horz_baseline))
            
            if (ir1 in range(nr)) and (ic1 in range(nc)) :
                
                if InitDisp[ir1,ic1] < D[ir,ic]:
                    InitDisp[ir1,ic1] = D[ir,ic]

                    Warped[ir1,ic1,:]  = np.squeeze(LFref[ir,ic,:])

    print("warped with ref " + str(iar0) + "," + str(iac0) + " for target " + str(iarT) + "," + str(iacT) )
    
    return Warped        

def computeMSE(Warped,LF_target, round_decimals = 4, occludes = None):
    
    LF_target = (255*LF_target).astype(np.uint8)
    nr,nc = LF_target.shape[0:2]          
    SumE = 0
    nr_sum = 0
    for ir in range(nr):
        for ic in range(nc):
            if occludes == None:
                valid_pixel_criterion = np.sum(Warped[ir,ic,:])>-1
            else:
                valid_pixel_criterion = not((ir,ic) in list(zip(occludes[0],occludes[1])))
            if valid_pixel_criterion:
#                for icomp in range(3):
                SumE  = SumE +  np.sum((Warped[ir,ic,:] - LF_target[ir,ic,:])**2)
                nr_sum = nr_sum + 1

    MSE = np.around(SumE/(nr_sum*3),decimals=round_decimals)
    return MSE

       
#def WarpingHCI_SOFT(LFref, D, target_view, ref_view, max_num_disps = 4, sigm = 1):
#
#    LFref = (255*LFref).astype(np.uint8)
#    nr,nc = LFref.shape[0:2]   
#    
#    iarT, iacT = target_view
#    iar0, iac0 = ref_view
#    
#    STORAGE = {}
#    
#    for ir in range(nr):
#        for ic in range(nc):
#            STORAGE[ir,ic] = {'fir':[],'fic':[],'colors':[],'disp':[]}
#
#    for ir in range(nr):    
#        for ic in range(nc):
#    
#            ir1 = ir-D[ir,ic]*(iarT-iar0)    
#            ic1 = ic-D[ir,ic]*(iacT-iac0)
#    
#            lir1 = np.floor(ir1).astype(int)
#            hir1 = np.ceil(ir1).astype(int)   
#            lic1 = np.floor(ic1).astype(int)
#            hic1 = np.ceil(ic1).astype(int)
#            
#            if lir1 in range(nr) and lic1 in range(nc) and hir1 in range(nr) and hic1 in range(nc): 
#    
#                Set = [(lir1, lic1), (lir1, hic1), (hir1, lic1), (hir1, hic1)]
#        
#                for irc, icc in Set:
#                
#                    STORAGE[irc,icc]['fir'].append(ir1-irc)       
#                    STORAGE[irc,icc]['fic'].append(ic1-icc)
#        
#                    colors = np.squeeze(LFref[ir,ic,:])
#        
#                    STORAGE[irc,icc]['colors'].append(np.transpose(colors))
#        
#                    STORAGE[irc,icc]['disp'].append(D[ir,ic])
#
#    SoftWarped = -1*np.ones((nr,nc,3),dtype=int)
#
#    for irc in range(nr):
#        for icc in range(nc):  
#         #   % pick the maximum disparity at which we reached (ir,ic)    
#        #    % and keep only the points within 3 values from that    
#            dispset = STORAGE[irc,icc]['disp']
#            
#            if len(dispset)>0:
#                inds = np.where(dispset > (max(dispset)-max_num_disps+1)) # say has size 4
##                num_disps = len(inds[0])
##            if np.prod(inds.shape) > 0:
#                firsmatter = np.array(STORAGE[irc,icc]['fir'])[inds] # size 4x1    
#                ficsmatter = np.array(STORAGE[irc,icc]['fic'])[inds] #% size 4x6
#    
#                eucl_dist = firsmatter**2+ficsmatter**2 # % size 4x1    
#    
#                weights = np.exp(-eucl_dist/sigm**2)[:,np.newaxis] # size 4x1
#    
#                colorsmatter = np.array(STORAGE[irc,icc]['colors'])[inds] # 4x3 array
#    
#                matcol = colorsmatter * np.tile(weights,(1,3))/sum(weights) #% 4x3
#    
#                finalcolors = sum(matcol)
#    
#                SoftWarped[irc,icc,:] = finalcolors
#                
#    return SoftWarped

#(LF0, D, target_view, ref_view, max_num_disps = 4, sigm = 1)
def WarpingHCI_SOFT2(LFref, D, target_view, ref_view, max_num_disps = 4, sigm = 1.4142, lower_disp_ratio = 0.05, large_number = 10**5, color_image=True, invalid_pix_val=-1):

    if D.dtype == np.uint8:        
        D = D.astype(np.float32)/255

    if color_image:            
        LFref = (255*LFref).astype(np.uint8)
        
    nr,nc,num_ch = LFref.shape

    
    iar0, iac0 = ref_view
    iarT, iacT = target_view
    
    # Initialize storage space
    firs = -large_number*np.ones((max_num_disps,nr,nc)) # neglected disparity fractional part in row direction
    fics = -large_number*np.ones((max_num_disps,nr,nc)) # neglected disparity fractional part in column direction
    colors = -large_number*np.ones((max_num_disps,nr,nc,num_ch)) # colors at storage element
    disps = -large_number*np.ones((max_num_disps,nr,nc)) # real valued disparity at storage element

    for ir in range(nr):
#    for ir in range(100):  #for debug
        for ic in range(nc):
            ir1 = ir-D[ir,ic]*(iarT-iar0)
            ic1 = ic-D[ir,ic]*(iacT-iac0)
            
            # Work only inside the (nr,nc) grid:
            lir1 = np.floor(ir1).astype(int)
            hir1 = np.ceil(ir1).astype(int)   
            lic1 = np.floor(ic1).astype(int)
            hic1 = np.ceil(ic1).astype(int)
            
            if lir1 in range(nr) and lic1 in range(nc) and hir1 in range(nr) and hic1 in range(nc): 
                
                # Candidate set for warping (ir,ic) to target location
                Set = [(lir1, lic1), (lir1, hic1), (hir1, lic1), (hir1, hic1)]
                # Go throuugh all candidates (irc,icc)
                for irc,icc in Set:
                
                    # best existent disparities
                    dispset = disps[:,irc,icc]
                    
                    #indvali en son bos slota set et.
                    indval = -1
                    for ij1 in range(max_num_disps):
                        if dispset[ij1]==-large_number:
                            indval = ij1                    
                        
                    if indval == -1: #eger hic bos slot yoksa,                     
                        valmin = large_number # valmin'i cok buyuk bir sayi olarak initialize et
                        indval1 = -1 # indval1 diye bir index initialize et
                        for ij1 in range(max_num_disps):
                                if dispset[ij1]<valmin:
                                    indval1 = ij1
                                    valmin = dispset[ij1]
    
                        if D[ir,ic] > valmin :
                            indval = indval1
#    
                    if indval > -1:
#                        % replace the component with index indval
                        firs[indval,irc,icc] = ir1-irc
                        fics[indval,irc,icc] = ic1-icc
                        colors[indval,irc,icc,:] = LFref[ir,ic,:]
                        disps[indval,irc,icc] =  D[ir,ic]

    if color_image:
        warped_type = int
    else:
        warped_type = LFref.dtype
    SoftWarped = invalid_pix_val*np.ones((nr,nc,num_ch),dtype=warped_type)

    for irc in range(nr):
        for icc in range(nc):            
#            % pick the maximum disparity at which we reached (ir,ic)
#            % and keep only the points within SB2 (e.g. =3)  from that
#            
            dispset = disps[:,irc,icc]
            max_dispset = max(dispset)
            
            if (max_dispset > -large_number):
                
                criterion = dispset > (max_dispset - abs(max_dispset)*lower_disp_ratio)
                inds = np.where(criterion)[0] # say has size 4                

                if len(inds) == 1:
                    SoftWarped[irc,icc,:] = colors[inds,irc,icc,:]
                else:                        
                    firsmatter = firs[inds,irc,icc] # % size 4x1
                    ficsmatter = fics[inds,irc,icc] # % size 4x6
                    eucl_dist = firsmatter**2+ficsmatter**2#; % size 4x1

                    weights = np.exp(-eucl_dist/sigm**2)[:,np.newaxis]#; % size 4x1
                    colorsmatter = colors[inds,irc,icc,:]#; % 4x3 array
                    matcol = colorsmatter * np.tile(weights,(1,num_ch))/sum(weights) #% 4x3
                    finalcolors = sum(matcol)
                    
                    SoftWarped[irc,icc,:] = finalcolors

    return SoftWarped                    

#def Warping_HCI_SOFT2_CRIT(LFref, D_in, target_view, ref_view, mu1, mu0, max_num_disps = 4, sigm = 1.4142, LFtarget=None):
#
#    lower_disp_ratio = 0.8
#    D = mu1*D_in + mu0
##% Initialize storage space
#    large_number = 10**5
#    LFref = (255*LFref).astype(np.uint8)
#    LFtarget = (255*LFtarget).astype(np.uint8)
#    nr,nc = LFref.shape[0:2]   
#    
#    SoftWarped = -1*np.ones((nr,nc,3),dtype=int)
#    
#    iar0, iac0 = ref_view
#    iarT, iacT = target_view    
##% Size of buffer
#    STORAGE = dict()
#    STORAGE["firs"] = -large_number*np.ones((max_num_disps,nr,nc)) # neglected disparity fractional part in row direction
#    STORAGE["fics"] = -large_number*np.ones((max_num_disps,nr,nc)) # neglected disparity fractional part in column direction
#    STORAGE["colors"] = -large_number*np.ones((max_num_disps,nr,nc,3)) # colors at storage element
#    STORAGE["disps"] = -large_number*np.ones((max_num_disps,nr,nc)) # real valued disparity at storage element
#    
#    STORAGE["ir"] = -1*np.ones((max_num_disps,nr,nc))
#    STORAGE["ic"] = -1*np.ones((max_num_disps,nr,nc))
#
#
#    STORAGE["firC"] =  np.zeros((max_num_disps,nr,nc))
#    STORAGE["firP"] =  np.zeros((max_num_disps,nr,nc))
#
#    STORAGE["ficC"] =  np.zeros((max_num_disps,nr,nc))
#    STORAGE["ficP"] = np.zeros((max_num_disps,nr,nc)) 
#    
#    CRIT = dict()
#
#    CRIT["Z"] = np.zeros((nr,nc,3))
#    CRIT["X"] = np.zeros((max_num_disps,nr,nc,3))
#    CRIT["isize"] = np.zeros((nr,nc),dtype=int)
#    CRIT["weightsi"] = np.zeros((max_num_disps,nr,nc))
#    CRIT["sumweights"] = np.zeros((nr,nc))
#    CRIT["firC"] = np.zeros((max_num_disps,nr,nc))
#    CRIT["firP"] = np.zeros((max_num_disps,nr,nc))
#    CRIT["ficC"] = np.zeros((max_num_disps,nr,nc))
#    CRIT["ficP"] = np.zeros((max_num_disps,nr,nc))
#    CRIT["ir"] = np.zeros((max_num_disps,nr,nc)) 
#    CRIT["ic"] = np.zeros((max_num_disps,nr,nc))
#
#
#
#    
#    for ir in range(nr):
#        for ic in range(nc):
#            ir1 = ir-D[ir,ic]*(iarT-iar0)
#            ic1 = ic-D[ir,ic]*(iacT-iac0)
#            
##            % Work only inside the (nr,nc) grid:
##            if( (ir1 >1) && (ir1<=nr-1) ...
##                    && (ic1 >1) && (ic1<=nc-1) )
##                % pause
##                % truncated target row index
##                lir1 = min(max(1,floor(ir1)),nr);
##                hir1 = min(max(1,ceil(ir1)),nr);
##                % truncated target column index
##                lic1 = min(max(1,floor(ic1)),nc);
##                hic1 = min(max(1,ceil(ic1)),nc);
#            lir1 = np.floor(ir1).astype(int)
#            hir1 = np.ceil(ir1).astype(int)   
#            lic1 = np.floor(ic1).astype(int)
#            hic1 = np.ceil(ic1).astype(int)         
#            if lir1 in range(nr) and lic1 in range(nc) and hir1 in range(nr) and hic1 in range(nc): 
#                Set = [(lir1, lic1), (lir1, hic1), (hir1, lic1), (hir1, hic1)]                
##                % Candidate set for warping (ir,ic) to target location
##                Set = [ lir1 lic1; lir1 hic1; hir1 lic1; hir1 hic1 ];
##                % Go throuugh all candidates (irc,icc)
#
#                for irc,icc in Set:                
##                    % best existent disparities
#                    dispset = STORAGE["disps"][:,irc,icc]
#
#                    valmax = np.max(dispset)
#                    if D[ir,ic] > valmax*lower_disp_ratio:
#                        indval = -1
#                        for ij1 in range(4):
#                            if dispset[ij1]==-large_number:
#                                indval = ij1
#                        
#                        if indval == -1:
#                            
#                            valmin = large_number
#                            indval1 = -1
#                            for ij1 in range(4):
#                                if dispset[ij1]<valmin:
#                                    indval1 = ij1
#                                    valmin = dispset[ij1]
#
#                            if D[ir,ic] > valmin :
#                                indval = indval1
#                                
#                                
#                        if indval > 0:
##                            % replace the component with index indval
#                            STORAGE["firs"][indval,irc,icc] = ir1-irc
#                            STORAGE["fics"][indval,irc,icc] = ic1-icc
#                            STORAGE["colors"][indval,irc,icc,:] = LFref[ir,ic,:]
#                            STORAGE["disps"][indval,irc,icc] =  D[ir,ic]
#                            
#                            STORAGE["ir"][indval,irc,icc] =  ir
#                            STORAGE["ic"][indval,irc,icc] =  ic
#                            STORAGE["firC"][indval,irc,icc] =  ir-irc
#                            STORAGE["firP"][indval,irc,icc] =  -(iarT-iar0)
#
#                            STORAGE["ficC"][indval,irc,icc] =  ic-icc
#                            STORAGE["ficP"][indval,irc,icc] =  -(iacT-iac0)
#
#     
#    for irc in range(nr):
#        for icc in range(nc):              
##            % pick the maximum disparity at which we reached (ir,ic)
##            % and keep only the points within SB2 (e.g. =3)  from that
#            
#            dispset = STORAGE["disps"][:,irc,icc]
#            max_dispset = max(dispset)
#            
#            if max_dispset > -large_number:
#                inds = np.where(dispset > (max(dispset)-max_num_disps+1))[0] # say has size 4                
#                isize = len(inds)
#                if isize>0:
#                    if isize == 1:
#                        SoftWarped[irc,icc,:] = STORAGE["colors"][inds,irc,icc,:]                    
#                    else:                    
#                        firsmatter = STORAGE["firs"][inds,irc,icc] #; % size 4x1
#                        ficsmatter = STORAGE["fics"][inds,irc,icc] #% size 4x6
#                        eucl_dist = firsmatter**2+ficsmatter**2#; #% size 4x1
#    
#                        weights = (1-eucl_dist/(sigm**2))[:,np.newaxis]#; % size 4x1
#                        colorsmatter = STORAGE["colors"][inds,irc,icc,:]#; % 4x3 array
#                        matcol = colorsmatter * np.tile(weights,(1,3))/sum(weights) #% 4x3
#                        
#                        finalcolors = sum( matcol)                    
#                        SoftWarped[irc,icc,:] = finalcolors
#    #                    for icolor = 1:3
#                        CRIT["Z"][irc,icc,:] = LFtarget[irc,icc,:]
#                        CRIT["X"][0:isize,irc,icc,:] = colorsmatter
#                        CRIT["isize"][irc,icc] = isize
#                        weightsi = 1-eucl_dist/(sigm**2)
#                        CRIT["weightsi"][0:isize,irc,icc] = weightsi
#                        CRIT["sumweights"][irc,icc] = sum(weightsi)
#                        CRIT["firC"][0:isize,irc,icc] = STORAGE["firC"][inds,irc,icc]
#                        CRIT["firP"][0:isize,irc,icc] = STORAGE["firP"][inds,irc,icc]
#                        CRIT["ficC"][0:isize,irc,icc] = STORAGE["ficC"][inds,irc,icc]
#                        CRIT["ficP"][0:isize,irc,icc] = STORAGE["ficP"][inds,irc,icc]
#                        CRIT["ir"][0:isize,irc,icc] = STORAGE["ir"][inds,irc,icc] 
#                        CRIT["ic"][0:isize,irc,icc] = STORAGE["ic"][inds,irc,icc]
#                    
#    return SoftWarped, CRIT

#def Warping_MultiTarget_CRIT(LF, Dref, target_views, ref_view,mu1,mu0):
#    
#    SoftWarpeds = dict()
#    CRITs = dict()
#    for target_view in target_views:
#        SoftWarpeds[target_view],CRITs[target_view] = Warping_HCI_SOFT2_CRIT(LF[ref_view], Dref, target_view, ref_view, LFtarget=LF[target_view], mu1=mu1, mu0=mu0)
#
#    return SoftWarpeds, CRITs

def WSOFT2_MultiTarget(LF, Dref, target_views, ref_view, max_num_disps = 4, sigm = 1.4142, lower_disp_ratio = 0.05, large_number = 10**5):

    Warpeds = dict()

    for target_view in target_views:
        print("doing target view: ")
        print(target_view)
        Warpeds[target_view] = WarpingHCI_SOFT2(LF[ref_view], Dref, target_view, ref_view,
                                                                max_num_disps = max_num_disps, sigm = sigm, 
                                                                lower_disp_ratio = lower_disp_ratio, large_number = large_number)

    return Warpeds
    
def computeMSE_MultiTarget(Warpeds, LF, target_views, max_value=255, occludes = None):
    '''
    returns dicts target_view >> MSE, target_view >> PSNR
    '''
    if occludes == None:
        occludes = dict()
        for target_view in target_views:
            occludes[target_view] = None
            
    MSEs, PSNRs = dict(), dict()
    for target_view in target_views:
        MSE = computeMSE(Warpeds[target_view], LF[target_view], occludes = occludes[target_view])
        PSNR = 10*np.log10(max_value**2/MSE)         
        MSEs[target_view],PSNRs[target_view] = MSE, PSNR
    MSEs["mean"],PSNRs["mean"] = np.mean(list(MSEs.values())), np.mean(list(PSNRs.values()))
    return MSEs,PSNRs

#def tf_computeMSE_MultiTarget(SoftWarpeds, LF, target_views, max_value=255):
#    
#    total_MSE = 0
#    for t,target_view in enumerate(target_views):
#        MSE = tf_computeMSE(SoftWarpeds[t], LF[target_view])
#        total_MSE = total_MSE + MSE
#    PSNR = 10*np.log10(max_value**2/total_MSE)    
#    return total_MSE, PSNR
#
#def tf_Warping_MultiTarget(LFref, Dref, target_views, ref_view):
#    
#
#    SoftWarpeds_list = []
#    for t, target_view in enumerate(target_views):
#        
#       SoftWarpeds_list.append(tf.py_func(fortf_WarpingHCI_SOFT3, [LFref, Dref, target_view, ref_view], tf.float32))
#
#    SoftWarpeds = tf.stack(SoftWarpeds_list)
#    return SoftWarpeds



def masker(ind_arr,desired_min, desired_max):
    ind_mask = np.zeros_like(ind_arr)
    ind_mask[np.where((ind_arr<=desired_max) * (ind_arr>=desired_min))] = 1
    masked_ind_arr = ind_mask * ind_arr
    return masked_ind_arr


def WarpingHCI_SOFT3(LFref, D, target_view, ref_view, max_num_disps = 4, sigm = 1.4142, lower_disp_ratio = 0.05, large_number = 10**5):
    '''
    LFref: numpy array float with shape nrxncx3
    D: numpy array with shape nrxnc (either float or uint8)
    returns Warped color image with the same shape as LFref (nr,nc,3)
    '''
    
    if D.dtype == np.uint8:        
        D = D.astype(np.float32)/255
        
    LFref = (255*LFref).astype(np.uint8)
    nr,nc = LFref.shape[0:2]   
    
    iar0, iac0 = ref_view
    iarT, iacT = target_view
    
    # Initialize storage space
    firs = -large_number*np.ones((max_num_disps+1,nr,nc)) # neglected disparity fractional part in row direction
    fics = -large_number*np.ones((max_num_disps+1,nr,nc)) # neglected disparity fractional part in column direction
    colors = -large_number*np.ones((max_num_disps+1,nr,nc,3)) # colors at storage element
    disps = -large_number*np.ones((max_num_disps+1,nr,nc)) # real valued disparity at storage element


    ir_arr = np.tile(np.arange(nr)[:,np.newaxis],(1,nc))  
    ic_arr = np.tile(np.arange(nc)[np.newaxis,:],(nr,1))              
    ir1_arr = ir_arr-D*(iarT-iar0)
    ic1_arr = ic_arr-D*(iacT-iac0)
    

            
    # Work only inside the (nr,nc) grid:
    lir1_arr = np.floor(ir1_arr).astype(int)
    masked_lir1_arr = masker(lir1_arr,0,nr-1)
    
    hir1_arr = np.ceil(ir1_arr).astype(int)   
    masked_hir1_arr = masker(hir1_arr,0,nr-1)
    
    lic1_arr = np.floor(ic1_arr).astype(int)
    masked_lic1_arr = masker(lic1_arr,0,nc-1)
    
    hic1_arr = np.ceil(ic1_arr).astype(int)
    masked_hic1_arr = masker(hic1_arr,0,nc-1)   
#    if lir1 in range(nr) and lic1 in range(nc) and hir1 in range(nr) and hic1 in range(nc): #MASK!!
    
    # Candidate set for warping (ir,ic) to target location
    Set = [(masked_lir1_arr, masked_lic1_arr), (masked_lir1_arr, masked_hic1_arr), 
           (masked_hir1_arr, masked_lic1_arr), (masked_hir1_arr, masked_hic1_arr)]
    # Go throuugh all candidates (irc,icc)

    for r_arr,c_arr in Set:
    
        # best existent disparities
        dispset = disps[:,r_arr,c_arr]

        indvals = -1*np.ones((nr,nc),int)
#        indval = -1'
        for ij1 in range(4):
            dispset_ij_equal_mlarge_bool = dispset[ij1]==-large_number
            indvals[dispset_ij_equal_mlarge_bool] = ij1
        
                  
        valmins = np.zeros((nr,nc))
        indval1s = -1*np.zeros((nr,nc),int)
        
        indval_equal_m1_bool = indvals == -1            
        valmins[indval_equal_m1_bool] == large_number
        indval1s[indval_equal_m1_bool] == -1
#        if indval == -1:     
                
 
        
        for ij1 in range(4):
#                if dispset[ij1]<valmin:
            dispset_smaller_valmin_bool = dispset[ij1]<valmins # if dispset[ij1]<valmin:
            
            indval1s[dispset_smaller_valmin_bool*indval_equal_m1_bool] = ij1
            
            valmins[dispset_smaller_valmin_bool*indval_equal_m1_bool] = dispset[ij1][dispset_smaller_valmin_bool*indval_equal_m1_bool]

        D_greater_valmins_bool = D>valmins              
        indvals[D_greater_valmins_bool*indval_equal_m1_bool] = indval1s[D_greater_valmins_bool*indval_equal_m1_bool]

        indval_greater_m1_bool = indvals > -1   
#                        % replace the component with index indval
        firs[(indvals+1)*indval_greater_m1_bool] = (ir1_arr-r_arr)
        fics[(indvals+1)*indval_greater_m1_bool] = (ic1_arr-c_arr)
        colors[(indvals+1)*indval_greater_m1_bool,:] = LFref
        disps[(indvals+1)*indval_greater_m1_bool] =  D

    SoftWarped = -1*np.ones((nr,nc,3),dtype=int)


    for irc in range(nr):
        for icc in range(nc):            
#            % pick the maximum disparity at which we reached (ir,ic)
#            % and keep only the points within SB2 (e.g. =3)  from that
#            
            dispset = disps[1:,irc,icc]
            max_dispset = max(dispset)
            
            if (max_dispset > -large_number):
                
                criterion = dispset > (max_dispset - abs(max_dispset)*lower_disp_ratio)
                inds = np.where(criterion)[0] # say has size 4                

                if len(inds) == 1:
                    SoftWarped[irc,icc,:] = colors[inds+1,irc,icc,:]
                else:                        
                    firsmatter = firs[inds+1,irc,icc] # % size 4x1
                    ficsmatter = fics[inds+1,irc,icc] # % size 4x6
                    eucl_dist = firsmatter**2+ficsmatter**2#; % size 4x1

                    weights = np.exp(-eucl_dist/sigm**2)[:,np.newaxis]#; % size 4x1
                    colorsmatter = colors[inds+1,irc,icc,:]#; % 4x3 array
                    matcol = colorsmatter * np.tile(weights,(1,3))/sum(weights) #% 4x3
                    finalcolors = sum(matcol)
                    
                    SoftWarped[irc,icc,:] = finalcolors
#    ##########################################        
#    dispset = disps[:,r_arr,c_arr]
#    max_dispset = np.max(dispset,0)
#    
#    
#    max_dispset_greater_very_low_bool = max_dispset > -large_number
#    tiled_max_dispset_greater_very_low_bool = np.tile(max_dispset_greater_very_low_bool[np.newaxis,:],(max_num_disps,1,1))
##    if (max_dispset > -large_number):
#        
#    criterion = dispset >= (max_dispset - np.abs(max_dispset)*lower_disp_ratio)
##    inds = np.where(criterion)[0] # say has size 4                
#
#    firsmatter = np.zeros_like(firs)  
#    ficsmatter = np.zeros_like(fics)                       
#    firsmatter[tiled_max_dispset_greater_very_low_bool*criterion] = firs[tiled_max_dispset_greater_very_low_bool*criterion] # % size 4x1
#    ficsmatter[tiled_max_dispset_greater_very_low_bool*criterion] = fics[tiled_max_dispset_greater_very_low_bool*criterion] # % size 4x6
#    eucl_dist = firsmatter**2+ficsmatter**2#; % size 4x1
#
#    weights = np.zeros_like(ficsmatter)
#    weights[tiled_max_dispset_greater_very_low_bool*criterion] = np.exp(-eucl_dist/sigm**2)[tiled_max_dispset_greater_very_low_bool*criterion]#[:,np.newaxis]#; % size 4x1
#    tiled_weights = np.tile(weights[:,:,:,np.newaxis],(1,1,1,3))
#    colorsmatter=np.zeros_like(colors)
#    colorsmatter[tiled_max_dispset_greater_very_low_bool*criterion,:] = colors[tiled_max_dispset_greater_very_low_bool*criterion,:]#; % 4x3 array
#    matcol = colorsmatter * tiled_weights/np.sum(tiled_weights,0) #% 4x3
#    SoftWarped= np.sum(matcol,0)
    
    return SoftWarped       



