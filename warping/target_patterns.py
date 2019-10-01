#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:21:38 2019

@author: emre
"""
import numpy as np

def visualize(iar_max,iac_max,target_views):
    T = np.zeros((iar_max,iac_max),dtype = int)
    for tv in target_views:
        T[tv] = 1
    print(T)
        
def NW_cross(ref_view,cross_size=9):
    iar0, iac0 = ref_view
    target_views = []

    
    for iac in range(iac0,iac0+cross_size):
        target_views.append((iar0,iac))
    
    for iar in range(iar0,iar0+cross_size):
        target_views.append((iar,iac0))
    
    for iar in range(iar0,iar0+cross_size):
        target_views.append((iar,iar+iac0-iar0))
    
    return target_views

def NE_cross(ref_view,cross_size=9):
    iar0, iac0 = ref_view
    target_views = []
    
    for iac in range(iac0-cross_size+1,iac0+1):
        target_views.append((iar0,iac))
    
    for iar in range(iar0,iar0+cross_size):
        target_views.append((iar,iac0))
    
    for iar in range(iar0,iar0+cross_size):
        target_views.append((iar,-iar+iac0+iar0))
        
    return target_views

def SW_cross(ref_view,cross_size=9):
    iar0, iac0 = ref_view
    target_views = []

    for iac in range(iac0,iac0+cross_size):
        target_views.append((iar0,iac))
    
    for iar in range(iar0-cross_size+1,iar0+1):
        target_views.append((iar,iac0))
    
    for iar in range(iar0-cross_size+1,iar0+1):
        target_views.append((iar,-iar+iac0+iar0))
    
    return target_views

def SE_cross(ref_view,cross_size=9):
    iar0, iac0 = ref_view
    target_views = []

    for iac in range(iac0-cross_size+1,iac0+1):
        if not (iac==iac0):         
            target_views.append((iar0,iac))
    
    for iar in range(iar0-cross_size+1,iar0+1):
        if not (iar==iar0):   
            target_views.append((iar,iac0))
    
    for iar in range(iar0-cross_size+1,iar0):     
        target_views.append((iar,iar))
    
    return target_views

def center_cross(ref_view,cross_size=9):
    iar0, iac0 = ref_view
    target_views = []    
    
    for iac in range(iac0 - cross_size//2, iac0 + cross_size//2 + 1):
        if not (iac==iac0): 
            target_views.append((iar0,iac))

    for iar in range(iar0 - cross_size//2, iar0 + cross_size//2 + 1):
        if not (iar==iar0): 
            target_views.append((iar,iac0))
        
    for iar in range(iar0 - cross_size//2, iar0 + cross_size//2 + 1):
        iac = iar - iar0 + iac0
        if not (iar==iar0 and iac==iac0): 
            target_views.append((iar,iac))

    for iar in range(iar0 - cross_size//2, iar0 + cross_size//2 + 1):
        iac = -iar +iar0 + iac0
        if not (iar==iar0 and iac==iac0): 
            target_views.append((iar, iac))        
        
        
    return target_views

def window(ref_view, window_size = 9):

    target_views = []
    iar0,iac0 = ref_view
    ws2 = window_size//2
    for iar in range(iar0-ws2, iar0+ws2 +1):
        target_views.append((iar,iac0-ws2))
        
    for iar in range(iar0-ws2, iar0+ws2 +1):
        target_views.append((iar,iac0+ws2))
        
    for iac in range(iac0-ws2+1, iac0+ws2):
        target_views.append((iar0-ws2,iac))

    for iac in range(iac0-ws2+1, iac0+ws2):
        target_views.append((iar0+ws2,iac))
    
    return target_views

def everyview(ref_view, lf_size = 9):
    target_views = []
    iar0,iac0 = ref_view
    for iar in range(lf_size):
        for iac in range(lf_size):
            if (iar,iac) != (iar0,iac0):
                target_views.append((iar,iac))    
    return target_views

def everyother(ref_views, lf_size = 9):
    target_views = []
    for iar in range(lf_size):
        for iac in range(lf_size):
            if (iar,iac) not in ref_views:
                target_views.append((iar,iac))    
    return target_views

def single_target(target_view):
    
    target_views = [target_view]
 
    return target_views



