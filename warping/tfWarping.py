#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:58:21 2019

@author: emre
"""
import tensorflow as tf

def true_fn1(ij1,dispset): 
    return ij1, dispset[ij1]

def false_fn1(indval1,valmin):
    return indval1.value(),valmin.value()
    


  
def computeMSE_MultiTarget(SoftWarpeds,LF,target_views):
    
    total_MSE = tf.constant(0)
    for target_view in target_views:
        MSE = computeMSE(SoftWarpeds[target_view], LF[target_view])
        total_MSE = total_MSE + MSE
    return total_MSE

def computeMSE(Warped,LF_target):
   
#    LF_target = (255*LF_target).astype(tf.uint8)
    nr,nc = LF_target.shape[0:2]          
    SumE = tf.constant(0)
    nr_sum = tf.constant(0)
    for ir in range(nr):
        for ic in range(nc):
            if tf.sum(Warped[ir,ic,:])>-1:
                SumE  = SumE +  tf.sum((Warped[ir,ic,:] - LF_target[ir,ic,:])**2)
                nr_sum = nr_sum + 1

    MSE = SumE/(nr_sum*3)
    return MSE
def WarpingHCI_SOFT2(LFref, Dref, target_view, ref_view, max_num_disps = 4, sigm = 1.4142, lower_disp_ratio = 0.05, large_number = 10**5):

    mu1 = tf.Variable(initial_value=1,dtype=tf.float32)
    mu0 = tf.Variable(initial_value=0,dtype=tf.float32)
    D = mu1*Dref + mu0    
#    LFref = (255*LFref).astype(np.uint8)
    nr,nc = LFref.shape[0:2]   
    
    iar0, iac0 = ref_view
    iarT, iacT = target_view
    
    # Initialize storage space
    firs = -large_number*tf.ones((max_num_disps,nr,nc)) # neglected disparity fractional part in row direction
    fics = -large_number*tf.ones((max_num_disps,nr,nc)) # neglected disparity fractional part in column direction
    colors = -large_number*tf.ones((max_num_disps,nr,nc,3)) # colors at storage element
    disps = -large_number*tf.ones((max_num_disps,nr,nc)) # real valued disparity at storage element

    for ir in range(nr):
#    for ir in range(100):  #for debug
        for ic in range(nc):
            ir1 = ir-D[ir,ic]*(iarT-iar0)
            ic1 = ic-D[ir,ic]*(iacT-iac0)
            
            # Work only inside the (nr,nc) grid:
            lir1 = tf.floor(ir1)
            hir1 = tf.ceil(ir1)   
            lic1 = tf.floor(ic1)
            hic1 = tf.ceil(ic1)
            
            if lir1 in range(nr) and lic1 in range(nc) and hir1 in range(nr) and hic1 in range(nc): 
                
                # Candidate set for warping (ir,ic) to target location
                Set = [(lir1, lic1), (lir1, hic1), (hir1, lic1), (hir1, hic1)]
                # Go throuugh all candidates (irc,icc)
                for irc,icc in Set:
                
                    # best existent disparities
                    dispset = disps[:,irc,icc]

                    indval = -1
                    for ij1 in range(4):
                        if dispset[ij1]==-large_number:
                            indval = ij1                    
                        
                    if indval == -1:                     
                        valmin = large_number
                        indval1 = -1
                        for ij1 in range(4):
                                if dispset[ij1]<valmin:
                                    indval1 = ij1
                                    valmin = dispset[ij1]
    
                        if D[ir,ic] > valmin :
                            indval = indval1
    
                    if indval > -1:
#                        % replace the component with index indval
                        firs[indval,irc,icc] = ir1-irc
                        fics[indval,irc,icc] = ic1-icc
                        colors[indval,irc,icc,:] = LFref[ir,ic,:]
                        disps[indval,irc,icc] =  D[ir,ic]

    SoftWarped = -1*np.ones((nr,nc,3),dtype=int)

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
                    matcol = colorsmatter * np.tile(weights,(1,3))/sum(weights) #% 4x3
                    finalcolors = sum(matcol)
                    
                    SoftWarped[irc,icc,:] = finalcolors

    return SoftWarped  
#def Warping_MultiTarget(LFref, Dref, target_views, ref_view):
#    
#    SoftWarpeds = dict()
#    for target_view in target_views:
#        SoftWarpeds[target_view] = WarpingHCI_SOFT2(LFref, Dref, target_view, ref_view)
#
#    return SoftWarpeds
#
#
#def WarpingHCI_SOFT2(LFref, Dref, target_view, ref_view, num_max_disps = 4, sigm = 1.414, large_number = 10**4):
#
#    
#    mu1 = tf.Variable(initial_value=1,dtype=tf.float32)
#    mu0 = tf.Variable(initial_value=0,dtype=tf.float32)
#    D = mu1*Dref + mu0
#    
##    LFref = (255*LFref).astype(tf.uint8)
#    nr,nc = LFref.shape[0:2]   
#    
#    iar0, iac0 = ref_view
#    iarT, iacT = target_view
#    
#    # Initialize storage space
#    firs = -large_number*tf.ones((num_max_disps,nr,nc)) # neglected disparity fractional part in row direction
#    fics = -large_number*tf.ones((num_max_disps,nr,nc)) # neglected disparity fractional part in column direction
#    colors = -large_number*tf.ones((num_max_disps,nr,nc,3)) # colors at storage element
#    disps = -large_number*tf.ones((num_max_disps,nr,nc)) # real valued disparity at storage element
#
#    for ir in range(nr):
#        for ic in range(nc):
#            ir1 = ir-D[ir,ic]*(iarT-iar0)
#            ic1 = ic-D[ir,ic]*(iacT-iac0)
#            
#            # Work only inside the (nr,nc) grid:
#            lir1 = tf.to_int32(tf.floor(ir1))
#            hir1 = tf.to_int32(tf.ceil(ir1)) 
#            lic1 = tf.to_int32(tf.floor(ic1))
#            hic1 = tf.to_int32(tf.ceil(ic1))
#            
##            if lir1 in range(nr) and lic1 in range(nc) and hir1 in range(nr) and hic1 in range(nc): 
#                
#            # Candidate set for warping (ir,ic) to target location
#            Set = [(lir1, lic1), (lir1, hic1), (hir1, lic1), (hir1, hic1)]
#            # Go throuugh all candidates (irc,icc)
#            for irc,icc in Set:
#            
#                # best existent disparities
#                dispset = disps[:,irc,icc]
#
#                indval = tf.Variable(-1,trainable=False)
#                for ij1 in range(4):
#                    if dispset[ij1]==-large_number:
#                        indval = ij1                    
#
#                def indval_equal_min1_true_fn():
#                    tf.control_dependencies([tf.assign(x, [2])])                
#                    valmin = tf.Variable(large_number,trainable=False,dtype=tf.float32)
#                    indval1 = tf.Variable(-1,trainable=False)
#                    for ij1 in range(4):
#                        indval1val,valminval = tf.cond(dispset[ij1]<valmin, lambda: true_fn1(ij1,dispset), lambda: false_fn1(indval1,valmin))
#                        indval1.assign(indval1val)
#                        valmin.assign(valminval)
#                        
#                    indvalval = tf.cond(D[ir,ic] > valmin, lambda: true_fn2(indval1), lambda: false_fn2(indval))    
#                    indval.assign(indvalval)
#
#                    
##                if indval == -1:   
#                tf.cond(indval == -1, indval_equal_min1_true_fn, lambda: false_fn)
#                    
#
#
#
#                if indval > -1:
##                        % replace the component with index indval
#                    firs[indval,irc,icc] = ir1-irc
#                    fics[indval,irc,icc] = ic1-icc
#                    colors[indval,irc,icc,:] = LFref[ir,ic,:]
#                    disps[indval,irc,icc] =  D[ir,ic]
#
#    SoftWarped = -1*tf.ones((nr,nc,3),dtype=tf.float32)
#
#    for irc in range(nr):
#        for icc in range(nc):            
##            % pick the maximum disparity at which we reached (ir,ic)
##            % and keep only the points within SB2 (e.g. =3)  from that
##            
#            dispset = disps[:,irc,icc]
#            max_dispset = tf.reduce_max(dispset)
#            
#            if (max_dispset > -large_number):
#
#                inds = tf.where(dispset > (max_dispset - tf.abs(max_dispset)*lower_disp_ratio))[0] # say has size 4                
#
#                if len(inds) == 1:
#                    SoftWarped[irc,icc,:] = colors[inds,irc,icc,:]
#                else:                        
#                    firsmatter = firs[inds,irc,icc] # % size 4x1
#                    ficsmatter = fics[inds,irc,icc] # % size 4x6
#                    eucl_dist = firsmatter**2+ficsmatter**2#; % size 4x1
#
#                    weights = tf.exp(-eucl_dist/sigm**2)[:,tf.newaxis]#; % size 4x1
#                    colorsmatter = colors[inds,irc,icc,:]#; % 4x3 array
#                    matcol = colorsmatter * tf.tile(weights,(1,3))/sum(weights) #% 4x3
#                    finalcolors = sum(matcol)
#                    
#                    SoftWarped[irc,icc,:] = finalcolors
#
#    return SoftWarped           
