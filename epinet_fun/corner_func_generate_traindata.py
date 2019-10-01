# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 10:32:22 2018

@author: Shin2018
"""
import numpy as np

def generate_traindata_for_train(traindata_all,traindata_labels,input_size,label_size,batch_size,Setting02_AngualrViews):
    
    """
     input: traindata_all   (16x512x512x9x9x3) uint8
            traindata_label (16x512x512x9x9)   float32
            input_size 23~   int
            label_size 1~    int
            batch_size 16    int
            Setting02_AngualrViews [0,1,2,3,4,5,6,7,8] for 9x9 
            boolmask_img4 (512x512)  bool // reflection mask for images[4]
            boolmask_img6 (512x512)  bool // reflection mask for images[6]
            boolmask_img15 (512x512) bool // reflection mask for images[15]


     Generate traindata using LF image and disparity map
     by randomly chosen variables.
     1.  gray image: random R,G,B --> R*img_R + G*img_G + B*imgB 
     2.  patch-wise learning: random x,y  --> LFimage[x:x+size1,y:y+size2]
     3.  scale augmentation: scale 1,2,3  --> ex> LFimage[x:x+2*size1:2,y:y+2*size2:2]

 
     output: traindata_batch_90d   (batch_size x input_size x input_size x len(Setting02_AngualrViews)) float32        
             traindata_batch_0d    (batch_size x input_size x input_size x len(Setting02_AngualrViews)) float32  
             traindata_batch_45d   (batch_size x input_size x input_size x len(Setting02_AngualrViews)) float32
             traindata_batch_m45d  (batch_size x input_size x input_size x len(Setting02_AngualrViews)) float32
             traindata_batch_label (batch_size x label_size x label_size )                   float32

    """
    
    
    
    """ initialize image_stack & label """ 
    traindata_batch_90d=np.zeros((batch_size,input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
    traindata_batch_0d=np.zeros((batch_size,input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
#    traindata_batch_45d=np.zeros((batch_size,input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
    traindata_batch_m45d=np.zeros((batch_size,input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)        
    
    traindata_batch_label=np.zeros((batch_size,label_size,label_size))
    
    
    
    
    """ inital variable """
    start1=Setting02_AngualrViews[0]
    end1=Setting02_AngualrViews[-1]    
    crop_half1=int(0.5*(input_size-label_size))
    
    
 
    
    """ Generate image stacks"""
    for ii in range(0,batch_size):
        sum_diff=0
        valid=0

        while( sum_diff<0.01*input_size*input_size  or  valid<1 ): 
             
            """//Variable for gray conversion//"""
            rand_3color=0.05+np.random.rand(3)
            rand_3color=rand_3color/np.sum(rand_3color) 
            R=rand_3color[0]
            G=rand_3color[1]
            B=rand_3color[2]
            
            
            

            aa_arr =np.array(range(traindata_all.shape[0]))      
            imid=np.random.choice(aa_arr)



            """
                //Shift augmentation for 7x7, 5x5 viewpoints,.. //
                Details in our epinet paper.
            """

            #randomly decide the scale##
            kk=np.random.randint(17)            
            if(kk<8):
                scale=1
            elif(kk<14):   
                scale=2
            elif(kk<17): 
                scale=3
            ############################
                
            idx_start = np.random.randint(0,512-scale*input_size)
            idy_start = np.random.randint(0,512-scale*input_size)    
            valid=1           
            """
                boolmask: reflection masks for images(4,6,15)
            """
#            if(imid==4 or 6 or 15):
#                if(imid==4):
#                    a_tmp=boolmask_img4
#                if(imid==6):
#                    a_tmp=boolmask_img6    
#                if(imid==15):
#                    a_tmp=boolmask_img15                            
#                    if( np.sum(a_tmp[idx_start+scale*crop_half1: idx_start+scale*crop_half1+scale*label_size:scale,
#                                     idy_start+scale*crop_half1: idy_start+scale*crop_half1+scale*label_size:scale])>0
#                         or np.sum(a_tmp[idx_start: idx_start+scale*input_size:scale, 
#                                         idy_start: idy_start+scale*input_size:scale])>0 ):
#                        valid=0
                    
            if(valid>0):      
#                seq0to8=np.array(Setting02_AngualrViews)#+ix_rd    
#                seq8to0=np.array(Setting02_AngualrViews[::-1])#+iy_rd
                
                image_center=(1/255)*np.squeeze(R*traindata_all[imid, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4, 4, 0].astype('float32')+
                                                G*traindata_all[imid, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4, 4, 1].astype('float32')+
                                                B*traindata_all[imid, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4, 4, 2].astype('float32'))
                sum_diff=np.sum(np.abs(image_center-np.squeeze(image_center[int(0.5*input_size),int(0.5*input_size)])))

                '''
                 Four image stacks are selected from LF full(512x512) images.
                 gray-scaled, cropped and scaled  
                 
                 traindata_batch_0d  <-- RGBtoGray( traindata_all[random_index, scaled_input_size, scaled_input_size, 4(center),    0to8    ] )
                 traindata_batch_90d   <-- RGBtoGray( traindata_all[random_index, scaled_input_size, scaled_input_size, 8to0,       4(center) ] )
                 traindata_batch_45d  <-- RGBtoGray( traindata_all[random_index, scaled_input_size, scaled_input_size, 8to0,         0to8    ] )
                 traindata_batch_m45d <-- RGBtoGray( traindata_all[random_index, scaled_input_size, scaled_input_size, 0to8,         0to8    ] )      
                 '''
                 
                corner_code = ["NW","NE","SW","SE"][np.random.randint(0,4)]
                corner_dict = {"NW":0,"NE":8,"SW":72,"SE":80}
                if corner_code == "NW":
                    rotm = 0
                    
                    iars_0d = 0
                    iacs_0d = list(range(0,9))
                    
                    iars_90d = list(range(0,9)[::-1])
                    iacs_90d = 0
            
                    iars_m45d = 0
                    iacs_m45d = 0
                    ksignr = 1
                    ksignc = 1
                    
                elif corner_code == "NE":
                    rotm = 1
                    
                    iars_0d = list(range(0,9))
                    iacs_0d = 8
                    
                    iars_90d = 0
                    iacs_90d = list(range(0,9))
            
                    iars_m45d = 0
                    iacs_m45d = 8                    
                    ksignr = 1
                    ksignc = -1
                    
                elif corner_code == "SW":
                    rotm = -1
                    
                    iars_0d = list(range(0,9))[::-1]
                    iacs_0d = 0
                    
                    iars_90d = 8
                    iacs_90d = list(range(0,9))[::-1]
            
                    iars_m45d = 8
                    iacs_m45d = 0        
                    ksignr = -1
                    ksignc = 1
                    
                elif corner_code == "SE":                    
                    rotm = 2
                    
                    iars_0d = 8
                    iacs_0d = list(range(0,9))[::-1]
                    
                    iars_90d = list(range(0,9))
                    iacs_90d = 8
            
                    iars_m45d = 8
                    iacs_m45d = 8                       
                    ksignr = -1
                    ksignc = -1 
        
                    
                traindata_batch_0d[ii,:,:,:]=np.rot90(np.squeeze(R*traindata_all[imid:imid+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, iars_0d, iacs_0d, 0].astype('float32')+
                                                         G*traindata_all[imid:imid+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, iars_0d,iacs_0d, 1].astype('float32')+
                                                         B*traindata_all[imid:imid+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, iars_0d, iacs_0d, 2].astype('float32')),rotm,(0,1))

#              
                traindata_batch_90d[ii,:,:,:]=np.rot90(np.squeeze(R*traindata_all[imid:imid+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, iars_90d, iacs_90d, 0].astype('float32')+
                                                        G*traindata_all[imid:imid+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, iars_90d, iacs_90d, 1].astype('float32')+
                                                        B*traindata_all[imid:imid+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, iars_90d, iacs_90d, 2].astype('float32')),rotm,(0,1))

                for kkk in range(start1,end1+1):
#                    

                    traindata_batch_m45d[ii,:,:,kkk-start1]=np.rot90(np.squeeze(R*traindata_all[imid:imid+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, ksignr*kkk + iars_m45d, ksignc*kkk + iacs_m45d, 0].astype('float32')+
                                                                        G*traindata_all[imid:imid+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, ksignr*kkk + iars_m45d, ksignc*kkk + iacs_m45d, 1].astype('float32')+
                                                                        B*traindata_all[imid:imid+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, ksignr*kkk + iars_m45d, ksignc*kkk + iacs_m45d, 2].astype('float32')),rotm,(0,1))
                '''
                 traindata_batch_label  <-- scale_factor*traindata_label[random_index, scaled_label_size, scaled_label_size] 
                 '''  
                traindata_batch_label[ii,:,:]=(1.0/scale)*np.rot90(traindata_labels[imid, idx_start+scale*crop_half1: idx_start+scale*crop_half1+scale*label_size:scale,
                                                                              idy_start+scale*crop_half1: idy_start+scale*crop_half1+scale*label_size:scale,corner_dict[corner_code]],rotm,(0,1))                 
#                if(len(traindata_labels.shape)==5):
#                    traindata_batch_label[ii,:,:]=(1.0/scale)*traindata_labels[imid, idx_start+scale*crop_half1: idx_start+scale*crop_half1+scale*label_size:scale,
#                                                                                  idy_start+scale*crop_half1: idy_start+scale*crop_half1+scale*label_size:scale,0,0]
#                else:
#                    traindata_batch_label[ii,:,:]=(1.0/scale)*traindata_labels[imid, idx_start+scale*crop_half1: idx_start+scale*crop_half1+scale*label_size:scale,
#                                                                                  idy_start+scale*crop_half1: idy_start+scale*crop_half1+scale*label_size:scale]
                                
    traindata_batch_90d=np.float32((1/255)*traindata_batch_90d)
    traindata_batch_0d =np.float32((1/255)*traindata_batch_0d)
#    traindata_batch_45d=np.float32((1/255)*traindata_batch_45d)
    traindata_batch_m45d=np.float32((1/255)*traindata_batch_m45d)
    
    return traindata_batch_90d,traindata_batch_0d,traindata_batch_m45d, traindata_batch_label  #,usage_check 


def data_augmentation_for_train(traindata_batch_90d, traindata_batch_0d,
                                traindata_batch_m45d, traindata_label_batchNxN, batch_size):
    """  
        For Data augmentation 
        (rotation, transpose and gamma)
        
    """ 
  
    for batch_i in range(batch_size):
        gray_rand=0.4*np.random.rand()+0.8
        
        traindata_batch_90d[batch_i,:,:,:]=pow(traindata_batch_90d[batch_i,:,:,:],gray_rand)
        traindata_batch_0d[batch_i,:,:,:]=pow(traindata_batch_0d[batch_i,:,:,:],gray_rand)
        traindata_batch_m45d[batch_i,:,:,:]=pow(traindata_batch_m45d[batch_i,:,:,:],gray_rand)               

        rotation_or_transp_rand=np.random.randint(0,2)    

        if rotation_or_transp_rand==1: 
            ##take the transpose of each view seperately:
            traindata_batch_90d_tmp6=np.copy(np.transpose(np.squeeze(traindata_batch_90d[batch_i,:,:,:]),(1, 0, 2)) )   
            traindata_batch_0d_tmp6=np.copy(np.transpose(np.squeeze(traindata_batch_0d[batch_i,:,:,:]),(1, 0, 2)) ) 
            traindata_batch_m45d_tmp6=np.copy(np.transpose(np.squeeze(traindata_batch_m45d[batch_i,:,:,:]),(1, 0, 2)) )

            ##take the transpose of the whole view grid:
            traindata_batch_0d[batch_i,:,:,:]=np.copy(traindata_batch_90d_tmp6[:,:,::-1])
            traindata_batch_90d[batch_i,:,:,:]=np.copy(traindata_batch_0d_tmp6[:,:,::-1])
            traindata_batch_m45d[batch_i,:,:,:]=np.copy(traindata_batch_m45d_tmp6)#[:,:,::-1])
            ##############################################
            traindata_label_batchNxN[batch_i,:,:]=np.copy(np.transpose(traindata_label_batchNxN[batch_i,:,:],(1, 0))) 
    

            


    return traindata_batch_90d, traindata_batch_0d,traindata_batch_m45d, traindata_label_batchNxN


def generate_traindata512(traindata_all,traindata_label,Setting02_AngualrViews,corner_code):
    """   
    Generate validation or test set( = full size(512x512) LF images) 
    
     input: traindata_all   (16x512x512x9x9x3) uint8
            traindata_label (16x512x512x9x9)   float32
            Setting02_AngualrViews [0,1,2,3,4,5,6,7,8] for 9x9            
     
    
     output: traindata_batch_90d   (batch_size x 512 x 512 x len(Setting02_AngualrViews)) float32        
             traindata_batch_0d    (batch_size x 512 x 512 x len(Setting02_AngualrViews)) float32  
             traindata_batch_45d   (batch_size x 512 x 512 x len(Setting02_AngualrViews)) float32
             traindata_batch_m45d  (batch_size x 512 x 512 x len(Setting02_AngualrViews)) float32
             traindata_label_batchNxN (batch_size x 512 x 512 )               float32            
    """

    corner_dict = {"NW":0,"NE":8,"SW":72,"SE":80}
    input_size=512; label_size=512;
    traindata_batch_90d=np.zeros((len(traindata_all),input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
    traindata_batch_0d=np.zeros((len(traindata_all),input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
    traindata_batch_m45d=np.zeros((len(traindata_all),input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)        
    
    traindata_label_batchNxN=np.zeros((len(traindata_all),label_size,label_size))
    
    """ inital setting """
    ### sz = (16, 27, 9, 512, 512) 

    crop_half1=int(0.5*(input_size-label_size))
    start1=Setting02_AngualrViews[0]
    end1=Setting02_AngualrViews[-1]
#        starttime=time.process_time() 0.375초 정도 걸림. i5 기준
    if corner_code == "NW":
        rotm = 0
        
        iars_0d = 0
        iacs_0d = list(range(0,9))
        
        iars_90d = list(range(0,9)[::-1])
        iacs_90d = 0

        iars_m45d = 0
        iacs_m45d = 0
        ksignr = 1
        ksignc = 1
        
    elif corner_code == "NE":
        rotm = 1
        
        iars_0d = list(range(0,9))
        iacs_0d = 8
        
        iars_90d = 0
        iacs_90d = list(range(0,9))

        iars_m45d = 0
        iacs_m45d = 8                    
        ksignr = 1
        ksignc = -1
        
    elif corner_code == "SW":
        rotm = -1
        
        iars_0d = list(range(0,9))[::-1]
        iacs_0d = 0
        
        iars_90d = 8
        iacs_90d = list(range(0,9))[::-1]

        iars_m45d = 8
        iacs_m45d = 0        
        ksignr = -1
        ksignc = 1
        
    elif corner_code == "SE":                    
        rotm = 2
        
        iars_0d = 8
        iacs_0d = list(range(0,9))[::-1]
        
        iars_90d = list(range(0,9))
        iacs_90d = 8

        iars_m45d = 8
        iacs_m45d = 8                       
        ksignr = -1
        ksignc = -1 
        
        
    for ii in range(0,len(traindata_all)):
        
        R = 0.299 ### 0,1,2,3 = R, G, B, Gray // 0.299 0.587 0.114
        G = 0.587
        B = 0.114

        imid = ii

        idx_start = 0
        idy_start = 0



        traindata_batch_0d[ii,:,:,:]=np.rot90(np.squeeze(R*traindata_all[imid:imid+1, idx_start: idx_start+input_size, idy_start: idy_start+input_size, iars_0d, iacs_0d, 0].astype('float32')+
                                                 G*traindata_all[imid:imid+1, idx_start: idx_start+input_size, idy_start: idy_start+input_size, iars_0d, iacs_0d, 1].astype('float32')+
                                                 B*traindata_all[imid:imid+1, idx_start: idx_start+input_size, idy_start: idy_start+input_size, iars_0d, iacs_0d, 2].astype('float32')),rotm,(0,1))
        
        traindata_batch_90d[ii,:,:,:]=np.rot90(np.squeeze(R*traindata_all[imid:imid+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, iars_90d, iacs_90d, 0].astype('float32')+
                                                G*traindata_all[imid:imid+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, iars_90d, iacs_90d, 1].astype('float32')+
                                                B*traindata_all[imid:imid+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, iars_90d, iacs_90d, 2].astype('float32')),rotm,(0,1))
        for kkk in range(start1,end1+1):
            

            traindata_batch_m45d[ii,:,:,kkk-start1]=np.rot90(np.squeeze(R*traindata_all[imid:imid+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, ksignr*kkk + iars_m45d, ksignc*kkk + iacs_m45d, 0].astype('float32')+
                                                                G*traindata_all[imid:imid+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, ksignr*kkk + iars_m45d, ksignc*kkk + iacs_m45d, 1].astype('float32')+
                                                                B*traindata_all[imid:imid+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, ksignr*kkk + iars_m45d, ksignc*kkk + iacs_m45d, 2].astype('float32')),rotm,(0,1))


        traindata_label_batchNxN[ii,:,:]=np.rot90(traindata_label[imid ,idx_start+crop_half1: idx_start+crop_half1+label_size,idy_start+crop_half1: idy_start+crop_half1+label_size, corner_dict[corner_code]],rotm,(0,1))
        
    traindata_batch_90d=np.float32((1/255)*traindata_batch_90d)
    traindata_batch_0d =np.float32((1/255)*traindata_batch_0d)
#    traindata_batch_45d=np.float32((1/255)*traindata_batch_45d)
    traindata_batch_m45d=np.float32((1/255)*traindata_batch_m45d)

    traindata_batch_90d=np.minimum(np.maximum(traindata_batch_90d,0),1)
    traindata_batch_0d=np.minimum(np.maximum(traindata_batch_0d,0),1)
#    traindata_batch_45d=np.minimum(np.maximum(traindata_batch_45d,0),1)
    traindata_batch_m45d=np.minimum(np.maximum(traindata_batch_m45d,0),1)

    return traindata_batch_90d,traindata_batch_0d,traindata_batch_m45d, traindata_label_batchNxN
##############################################################
##############################################################
if __name__ == "__main__":
        
    #traindata_all = np.ones((16,512,512,9,9,3), dtype=np.uint8)
    #traindata_label = np.ones((16,512,512,9,9), dtype=np.float32)
    input_size=25
    label_size=input_size-22
    batch_size=1
    Setting02_AngualrViews = [0,1,2,3,4,5,6,7,8]
    nviews = len(Setting02_AngualrViews)
    #boolmask_img4 = np.ndarray((512,512),np.bool)
    #boolmask_img6 = np.ndarray((512,512),np.bool)
    #boolmask_img15 = np.ndarray((512,512),np.bool)
    #
    #(traindata_batch_90d, traindata_batch_0d,
    # traindata_batch_45d, traindata_batch_m45d, 
    # traindata_label_batchNxN)= generate_traindata_for_train(traindata_all,traindata_label,
    #                                                         input_size,label_size,batch_size,
    #                                                         Setting02_AngualrViews,
    #                                                         boolmask_img4,boolmask_img6,boolmask_img15)                
    
    #(traindata_batch_90d, traindata_batch_0d,
    # traindata_batch_45d,traindata_batch_m45d, 
    # traindata_label_batchNxN) =  data_augmentation_for_train(traindata_batch_90d, 
    #                                                          traindata_batch_0d,
    #                                                          traindata_batch_45d,
    #                                                          traindata_batch_m45d, 
    #                                                          traindata_label_batchNxN,
    #                                                          batch_size) 
    from matplotlib import pyplot as plt
    import cv2
    
    dog=plt.imread('C:\\Local\\vhemka\\Data\\dog.png')
    dog = cv2.cvtColor(cv2.resize(dog,(input_size,input_size)),cv2.COLOR_RGB2GRAY)
    plt.imshow(dog)
    plt.show()
    traindata_batch_90d = np.tile(dog[np.newaxis,:,:,np.newaxis],(1,1,1,nviews))
    ##########################################################
    cat=plt.imread('C:\\Local\\vhemka\\Data\\cat.png')
    cat = cv2.cvtColor(cv2.resize(cat,(input_size,input_size)),cv2.COLOR_RGB2GRAY)
    plt.imshow(cat)
    plt.show()
    traindata_batch_0d = np.tile(cat[np.newaxis,:,:,np.newaxis],(1,1,1,nviews))
    ##########################################################
    car=plt.imread('C:\\Local\\vhemka\\Data\\car.png')
    car = cv2.cvtColor(cv2.resize(car,(input_size,input_size)),cv2.COLOR_RGB2GRAY)
    plt.imshow(car)
    plt.show()
    traindata_batch_45d = np.tile(car[np.newaxis,:,:,np.newaxis],(1,1,1,nviews))
    ##########################################################
    house=plt.imread('C:\\Local\\vhemka\\Data\\house.png')
    house = cv2.cvtColor(cv2.resize(house,(input_size,input_size)),cv2.COLOR_RGB2GRAY)
    plt.imshow(house)
    plt.show()
    traindata_batch_m45d = np.tile(house[np.newaxis,:,:,np.newaxis],(1,1,1,nviews))
    

    
    
    traindata_label_batchNxN=np.zeros((batch_size,label_size,label_size))
    
    """  
        For Data augmentation 
        (rotation, transpose and gamma)
        
    """ 
      
    #for batch_i in range(batch_size):
    batch_i=0
    gray_rand=0.4*np.random.rand()+0.8
    
    traindata_batch_90d[batch_i,:,:,:]=pow(traindata_batch_90d[batch_i,:,:,:],gray_rand)
    traindata_batch_0d[batch_i,:,:,:]=pow(traindata_batch_0d[batch_i,:,:,:],gray_rand)
    traindata_batch_45d[batch_i,:,:,:]=pow(traindata_batch_45d[batch_i,:,:,:],gray_rand)
    traindata_batch_m45d[batch_i,:,:,:]=pow(traindata_batch_m45d[batch_i,:,:,:],gray_rand)               
    

    
    ##take the transpose of each view seperately:
    traindata_batch_90d_tmp6=np.copy(np.flip(np.squeeze(traindata_batch_90d[batch_i,:,:,:]),(1, 0, 2)) )   
    traindata_batch_0d_tmp6=np.copy(np.flip(np.squeeze(traindata_batch_0d[batch_i,:,:,:]),(1, 0, 2)) ) 
    traindata_batch_m45d_tmp6=np.copy(np.flip(np.squeeze(traindata_batch_m45d[batch_i,:,:,:]),(1, 0, 2)) )

    ##take the transpose of the whole view grid:
    traindata_batch_0d[batch_i,:,:,:]=np.copy(traindata_batch_90d_tmp6[:,:,::-1])
    traindata_batch_90d[batch_i,:,:,:]=np.copy(traindata_batch_0d_tmp6[:,:,::-1])
    traindata_batch_m45d[batch_i,:,:,:]=np.copy(traindata_batch_m45d_tmp6)#[:,:,::-1])
    
    traindata_label_batchNxN[batch_i,:,:]=np.copy(np.transpose(traindata_label_batchNxN[batch_i,:,:],(1, 0))) 







