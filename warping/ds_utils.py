#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 18:11:20 2019

@author: emre
"""
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("/home/emre/Documents/kodlar/epinet-master/epinet_fun/")
from func_pfm import read_pfm

def readGTs(iar_max, iac_max, gt_dir, sample, disp_or_depth):

    Ds = dict()

        
    for iar0 in range(iar_max):
        for iac0 in range(iac_max):
            index0 = (iar0)*iac_max+iac0
            filename = gt_dir + sample + '/gt_'+disp_or_depth+'_lowres_Cam' + str(index0).zfill(3) + '.pfm'
            D = read_pfm(filename)
            Ds[iar0,iac0] = D
    return Ds


def readLF(iar_max, iac_max, nr, nc, DIR_TEST, sample, vis):
    LF0 = np.zeros((iar_max, iac_max, nr, nc, 3))
    
    if vis:
        fig,ax = plt.subplots()
        
    for iar in range(iar_max):
        for iac in range(iac_max): 
    

            index1 = (iar)*iac_max+iac
            AA = plt.imread(DIR_TEST + sample + '/input_Cam' + str(index1).zfill(3) + '.png')
            
            if vis:
                ax.imshow(AA)
                plt.show()
                
            LF0[iar,iac,:,:,:] = np.double(AA)
            
    return LF0