# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:54:06 2018

@author: shinyonsei2
"""
from tensorflow.contrib.keras.api.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.models import Model, Sequential
from tensorflow.contrib.keras.api.keras.layers import Input , Activation
from tensorflow.contrib.keras.api.keras.layers import Conv2D, Reshape
from tensorflow.contrib.keras.api.keras.layers import Dropout,BatchNormalization
from tensorflow.contrib.keras.api.keras.layers import concatenate

feats = []
feat_names = []
def layer1_multistream(input_dim1,input_dim2,input_dim3,filt_num,do_vis=False,name=None):    
    seq = Sequential()
    ''' Multi-Stream layer : Conv - Relu - Conv - BN - Relu  '''
    
    if do_vis:
        global feats
        global feat_names

#    seq.add(Reshape((input_dim1,input_dim12,input_dim3),input_shape=(input_dim1, input_dim2, input_dim3,1)))
    for i in range(3):
        seq.add(Conv2D(int(filt_num),(2,2),input_shape=(input_dim1, input_dim2, input_dim3), padding='valid', name='S1_c1%d' %(i) ))
        seq.add(Activation('relu', name='S1_relu1%d' %(i))) 
        if do_vis: #and i == 0:
            feats.append(seq)
            feat_names.append(name + '_S1_c1%d' %(i))
        seq.add(Conv2D(int(filt_num),(2,2), padding='valid', name='S1_c2%d' %(i) )) 
        seq.add(BatchNormalization(axis=-1, name='S1_BN%d' % (i)))
        seq.add(Activation('relu', name='S1_relu2%d' %(i))) 
        if do_vis:
            feats.append(seq)
            feat_names.append(name + '_S1_c2%d' %(i))
    seq.add(Reshape((input_dim1-6,input_dim2-6,int(filt_num))))

    return seq
def layer2_merged(input_dim1,input_dim2,input_dim3,filt_num,conv_depth):
    ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
    
    seq = Sequential()
    
    for i in range(conv_depth):
        seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S2_c1%d' % (i) ))
        seq.add(Activation('relu', name='S2_relu1%d' %(i))) 
        seq.add(Conv2D(filt_num,(2,2), padding='valid', name='S2_c2%d' % (i))) 
        seq.add(BatchNormalization(axis=-1, name='S2_BN%d' % (i)))
        seq.add(Activation('relu', name='S2_relu2%d' %(i)))
          
    return seq     

def layer3_last(input_dim1,input_dim2,input_dim3,filt_num):   
    ''' last layer : Conv - Relu - Conv ''' 
    
    seq = Sequential()
    
    for i in range(1):
        seq.add(Conv2D(filt_num,(2,2), padding='valid',input_shape=(input_dim1, input_dim2, input_dim3), name='S3_c1%d' %(i) )) # pow(25/23,2)*12*(maybe7?) 43 3
        seq.add(Activation('relu', name='S3_relu1%d' %(i)))
        
    seq.add(Conv2D(1,(2,2), padding='valid', name='S3_last')) 

    return seq 

def define_cepinet(sz_input,sz_input2,view_n,conv_depth,filt_num,learning_rate,for_vis = False):
    global feats
    if for_vis:
        feats = []
    else:
        feats = None
    ''' 4-Input : Conv - Relu - Conv - BN - Relu ''' 
    input_stack_90d = Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_90d')
    input_stack_0d= Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_0d')
#    input_stack_45d= Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_45d')
    input_stack_M45d= Input(shape=(sz_input,sz_input2,len(view_n)), name='input_stack_M45d')
    num_stacks = 3
    with tf.variable_scope("4-Stream"):
        ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu ''' 
        mid_90d = layer1_multistream(sz_input,sz_input2,len(view_n),int(filt_num),do_vis=True,name="90d")(input_stack_90d)
        mid_0d = layer1_multistream(sz_input,sz_input2,len(view_n),int(filt_num),do_vis=True,name="0d")(input_stack_0d)    
    #    mid_45d=layer1_multistream(sz_input,sz_input2,len(view_n),int(filt_num))(input_stack_45d)    
        mid_M45d = layer1_multistream(sz_input,sz_input2,len(view_n),int(filt_num),do_vis=True,name="M45d")(input_stack_M45d)   

    with tf.variable_scope("Merge"):
        ''' Merge layers ''' 
        mid_merged = concatenate([mid_90d,mid_0d,mid_M45d],  name='mid_merged')
        
        ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
        mid_merged_=layer2_merged(sz_input-6,sz_input2-6,int(num_stacks*filt_num),int(num_stacks*filt_num),conv_depth)(mid_merged)

    with tf.variable_scope("Last"):
        ''' Last Conv layer : Conv - Relu - Conv '''
        output=layer3_last(sz_input-18,sz_input2-18,int(num_stacks*filt_num),int(num_stacks*filt_num))(mid_merged_)

    if for_vis:
        feat_outs90d = [feat(input_stack_90d) for feat in feats[0:6]]
        feat_outs0d =  [feat(input_stack_0d) for feat in feats[6:12]]
        feat_outsM45d = [feat(input_stack_M45d) for feat in feats[12:18]]
        outputs = feat_outs90d + feat_outs0d + feat_outsM45d + [output]

    else: 
        outputs = [output]
    model_512 = Model(inputs = [input_stack_90d,input_stack_0d,
#                               input_stack_45d,
                               input_stack_M45d], outputs = outputs)
    opt = RMSprop(lr=learning_rate)
    model_512.compile(optimizer=opt, loss='mae')
    model_512.summary() 
    
    return model_512, feat_names