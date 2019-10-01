from matplotlib import pyplot as plt
import numpy as np

from WarpingHCI import warp
from ds_utils import readLF, readGTs
import os,sys
import time
from shutil import copyfile
import inspect

## Config#############


DIR_TEST = '/media/emre/Data/heidelberg_full_data/additional/'
gt_dir = '/media/emre/Data/heidelberg_full_data/additional_depth_disp_all_views/'
samples = os.listdir(gt_dir)

visLF = True
iar_max = 9
iac_max = 9

nr = 512
nc = 512
gamma = 0.7
computeMSE = False

######################
def list_ref_views(target_view):
    
    tr,tc = target_view
    
    ref_rs = tuple(set((int(np.floor(tr)), int(np.ceil(tr)))))
    ref_cs = tuple(set((int(np.floor(tc)), int(np.ceil(tc)))))
    
    ref_views = []
    for ref_r in ref_rs:
        for ref_c in ref_cs:
            ref_views.append((ref_r,ref_c))   

    return ref_views
    
    
def refine_tvs(target_views):
    new_target_views = []
    for target_view in target_views:
        rt,ct = target_view
        if not ( (rt - int(rt) == 0) and (ct - int(ct) == 0) ):
            if rt<=(iar_max-1) and 0<=rt and ct<=(iac_max-1) and 0<=ct:
                new_target_views.append(target_view)
    return new_target_views

cc = (iar_max-1)//2
#horz_views = list(zip(4*np.ones(iar_max,np.int),range(iar_max))) ## 0d
vert_views_r = list((cc-np.arange(0,cc,gamma))*10/10)[::-1] + list(np.arange(cc+gamma,2*cc,gamma)*10/10)
stack_len = len(vert_views_r)
vert_views_c = 4*np.ones(stack_len)
vert_views = list(zip(vert_views_r,vert_views_c)) ## 90d 
#####################################
angM45_r = vert_views_r
angM45_c = list(range(cc-(stack_len-1)//2,cc+(stack_len+1)//2))
angM45 = list(zip(angM45_r,angM45_c)) ## M45d                 
########################################
ang45_r = vert_views_r
ang45_c = angM45_c[::-1]
ang45 = list(zip(ang45_r,ang45_c)) ## 45d 

target_views = list(set(vert_views + angM45 + ang45))
## get rid of integer views
target_views = refine_tvs(target_views)
#####
print ("target views: " + str(target_views))
plt.title("target view locations")
plt.scatter(list(zip(*target_views))[1],list(zip(*target_views))[0])

#root_dir = "/home/emre/Documents/kodlar/tabus_matlab/"
output_dir = "/media/emre/Data/warpings/" + time.strftime("%d-%m-%H-%M") + "_gamma_"+str(gamma)+ "/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
copyfile(curr_file,output_dir + curr_file.split("/")[-1])    
       
#########################################################
for sample in samples:
    LF0 = readLF(iar_max, iac_max, nr, nc, DIR_TEST, sample, vis = visLF)
    
    Ds = readGTs(iar_max, iac_max, gt_dir, sample)
    
    #################################################################
    #### WARPING #########################################
    allMSEs = dict()
    allWarpeds = dict()
    #for ref_view in ref_views:
    for target_view in target_views:
    
        targets_ref_views = list_ref_views(target_view)
    
        print("warping for target: " + str(target_view[0]) + "," + str(target_view[1]))
        allMSEs[target_view], allWarpeds[target_view] = warp(targets_ref_views, target_view, Ds, LF0, computeMSE)
     
    np.save(output_dir + "Warped_"+sample+".npy",allWarpeds)
    ########################################################        
            



 
 
