#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:45:30 2019

@author: emre
"""

from matplotlib import pyplot as plt
import numpy as np
from glob import glob
from warping_usefuls import extract_PSNRs

### CONFIGURATION ####
LF_shape = (9,9)
crnr_coord = (0,8)
sample  = 'boxes'
corner_warp_dir = '/media/emre/Data/cepinet_runs/08-05-12-40/tests/0034__10-05-15-19__all_corners__hci/pfms/corner_warping_14-05-15-13/'
center_warp_dir = '/media/emre/Data/epinet_runs/09-05-13-05/tests/14-05-15-21__hci/pfms/center_warping_14-05-15-24/'
######################

corner_dir = glob(corner_warp_dir + 'corner_warp_'+sample+'_ref_'+str(crnr_coord[0])+'_'+str(crnr_coord[1])+'_*')[0]+'/'
corner_impaths = glob(corner_dir + '*.png')
corner_PSNRs = extract_PSNRs(corner_impaths)

center_dir = glob(center_warp_dir +sample+'_ref_4_4')[0]+'/'
center_impaths = glob(center_dir+'*.png')
center_PSNRs = extract_PSNRs(center_impaths)

diff_PSNRs = corner_PSNRs - center_PSNRs
diff_PSNRs[crnr_coord] = 0
diff_PSNRs[4,4] = 0
#im = plt.imread(impath)[:,:,0:3]
#
#occl_mask = np.sum(im,-1)==0