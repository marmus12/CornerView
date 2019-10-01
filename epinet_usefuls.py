#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:59:19 2019

@author: emre
"""

def infer_cos_from_corner(corner_coords,corner_code,an,model_cw):
 
    corner_x, corner_y = corner_coords
    
    if corner_code == "NW":
        center_x = corner_x + (model_cw-1)//2
        center_y = corner_y + (model_cw-1)//2  

    elif corner_code == "NE":
        center_x = corner_x + (model_cw-1)//2
        center_y = corner_y - (model_cw-1)//2      
        
    elif corner_code == "SW":
        center_x = corner_x - (model_cw-1)//2
        center_y = corner_y + (model_cw-1)//2      
        
    elif corner_code == "SE":
        center_x = corner_x - (model_cw-1)//2
        center_y = corner_y - (model_cw-1)//2   
          
    center_os_x = center_x - (an - 1)//2
    center_os_y = center_y - (an - 1)//2    
    center_offset = (center_os_x, center_os_y)
    
    return center_offset

def infer_cos_from_cc(center_coord,ds):
    
    iar, iac = center_coord
    iar0, iac0 = ds.iar_max//2, ds.iac_max//2
    
    center_offset = (iar - iar0, iac - iac0) 
    return center_offset


def infer_center_coords(ds,center_offset):
    
    co_r, co_c = center_offset
    iar, iac = (ds.iar_max-1)//2 + co_r, (ds.iac_max-1)//2 + co_c 
    return iar, iac