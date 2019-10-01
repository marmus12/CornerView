# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:54:06 2018

@author: shinyonsei2
"""
#from tensorflow.contrib.keras.api.keras.optimizers import RMSprop
#
#from tensorflow.contrib.keras.api.keras.models import Model, Sequential
#from tensorflow.contrib.keras.api.keras.layers import Input , Activation
#from tensorflow.contrib.keras.api.keras.layers import Conv2D, Reshape
#from tensorflow.contrib.keras.api.keras.layers import Dropout,BatchNormalization
#from tensorflow.contrib.keras.api.keras.layers import concatenate
import tensorflow as tf


def layer1_multistream(inputt, filt_num):    

    ''' Multi-Stream layer : Conv - Relu - Conv - BN - Relu  '''
    curr_input = inputt

    for i in range(3):
        output = tf.layers.conv2d(curr_input, filters=filt_num,kernel_size=(2,2),name='S1_c1%d' %(i))        
#        seq.add(Conv2D(int(filt_num),(2,2),input_shape=(input_dim1, input_dim2, input_dim3), padding='valid', name='S1_c1%d' %(i) ))
        output = tf.nn.relu(output)
#        seq.add(Activation('relu', name='S1_relu1%d' %(i))) 
        output = tf.layers.conv2d(curr_input, filters=filt_num,kernel_size=(2,2),name='S1_c2%d' %(i))                
#        seq.add(Conv2D(int(filt_num),(2,2), padding='valid', name='S1_c2%d' %(i) )) 
        output = tf.layers.batch_normalization(output,axis=-1,name='S1_BN%d' % (i))        
#        seq.add(BatchNormalization(axis=-1, name='S1_BN%d' % (i)))
        curr_input = tf.nn.relu(output)    
#        seq.add(Activation('relu', name='S1_relu2%d' %(i))) 
    return curr_input 

def layer2_merged(inputt, filt_num, conv_depth):
    ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
    curr_input = inputt    
    
    for i in range(conv_depth):
        output = tf.layers.conv2d(curr_input, filters=filt_num,kernel_size=(2,2),name='S2_c1%d' %(i))  
#        seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
        output = tf.nn.relu(output) 
#        seq.add(Activation('relu', name='S2_relu1%d' %(i))) 
        output = tf.layers.conv2d(output, filters=filt_num,kernel_size=(2,2),name='S2_c2%d' %(i))          
#        seq.add(Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i))) 
        output = tf.layers.batch_normalization(output,axis=-1,name='S2_BN%d' % (i))           
#        seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
        curr_input = tf.nn.relu(output)         
#        seq.add(Activation('relu', name='S2_relu2%d' %(i)))
          
    return curr_input   

def layer3_last(inputt,filt_num):   
    ''' last layer : Conv - Relu - Conv ''' 
    output = tf.layers.conv2d(inputt, filters=filt_num,kernel_size=(2,2),name='S3_c10')  
#    seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S3_c1%d' %(0) )) # pow(25/23,2)*12*(maybe7?) 43 3
    output = tf.nn.relu(output) 
#    seq.add(Activation('relu', name='S3_relu1%d' %(i)))
    output = tf.layers.conv2d(output, filters=filt_num, kernel_size=(2,2),name='S3_last')          
#    seq.add(Conv2D(1,(2,2), padding='valid', name='S3_last')) 
    return output

def define_epinet(inputts, conv_depth, filt_num):


    ''' 4-Input : Conv - Relu - Conv - BN - Relu ''' 
    inputt_90d, inputt_0d, inputt_45d, inputt_M45d = inputts    
#    input_stack_90d = Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_90d')
#    input_stack_0d= Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_0d')
#    input_stack_45d= Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_45d')
#    input_stack_M45d= Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_M45d')
#    
    ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu ''' 
    mid_90d=layer1_multistream(inputt_90d, filt_num)
    mid_0d=layer1_multistream(inputt_0d, filt_num)    
    mid_45d=layer1_multistream(inputt_45d, filt_num)    
    mid_M45d=layer1_multistream(inputt_M45d, filt_num)  

    ''' Merge layers ''' 
    mid_merged = concatenate([mid_90d,mid_0d,mid_45d,mid_M45d],  name='mid_merged')
    
    ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
    mid_merged_=layer2_merged(sz_input-6,sz_input2-6,int(4*filt_num),int(4*filt_num),conv_depth)(mid_merged)

    ''' Last Conv layer : Conv - Relu - Conv '''
    output=layer3_last(sz_input-18,sz_input2-18,int(4*filt_num),int(4*filt_num))(mid_merged_)

    
    return output