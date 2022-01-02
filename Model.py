# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 12:48:27 2021

@author: s1253
"""

import numpy as np
from tensorflow.keras import Model

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Activation, BatchNormalization,\
    Dropout, UpSampling2D, LayerNormalization,  Embedding, Concatenate, add, AveragePooling2D




def conv_norm(inp_x, filter_size, norm, kernel_size, strides=(1, 1)):
    
    
    if norm=="batch_norm":
        x = Conv2D(filter_size, kernel_size, strides=strides, padding="same")(inp_x)
        x = BatchNormalization()(x)
        x = Activation('leaky_relu')(x)

    else:
        x = Conv2D(filter_size, kernel_size, strides=strides, padding="same")(inp_x)
        x = LayerNormalization()(x)
        x = Activation('leaky_relu')(x)
        
        
    return x

def resBlock(inp_x, filter_size, kernel_size, strides, short_cut=0, norm="batch_norm"):
    
    x = conv_norm(inp_x, filter_size, kernel_size=kernel_size, strides=strides, norm=norm)
    x = conv_norm(x, filter_size, kernel_size=kernel_size, norm=norm)
    
    
    
    if short_cut==1:
        inp = conv_norm(inp_x, filter_size, kernel_size=(1, 1), norm=norm)
        
        out = add([x, inp])
        
        return out
    else:
        
        out = add([x, inp_x])
        
        return out
    

    
class Generator():
    def __init__(self,
                 input_dim,
                 label_len,
                 generator_shape,
                 batch_norm,
                 dropout,
                 generator_upsample,
                 generator_conv_filters,
                 generator_conv_kernal,
                 generator_short_cut,
                 generator_conv_stride=[1,1,1,1,1,1]
                 ):
        
        self.label_len = label_len
        self.input_dim = input_dim
        self.n_layer = len(generator_conv_filters)
        self.generator_shape = generator_shape
        self.batch_norm = batch_norm
        
        self.dropout = dropout
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernal = generator_conv_kernal
        self.generator_conv_stride = generator_conv_stride
        self.generator_short_cut = generator_short_cut
    
    def get_label(self, label_inp, label_len):
        
        x = Embedding(label_len, self.input_dim)(label_inp)
        x = Dense(np.prod((4, 4, 1)))(x)
        x = Reshape((4, 4, 1))(x)
        return x
        
        
    
    def build_layer(self):
        
        label_inp = Input((1,))
        generator_input = Input(self.input_dim, name="generator_input")
        
        label = self.get_label(label_inp, self.label_len)
        
        x = Dense(np.prod(self.generator_shape))(generator_input)
        
        x = Reshape(self.generator_shape)(x)
        
        x = Concatenate()([x, label])
        
        if self.batch_norm:
            x = BatchNormalization()(x)
        
        x = Activation('leaky_relu', name="relu")(x)
        
        
        
        
        
        if self.dropout:
            x = Dropout(self.dropout)(x)
          
        for i in range(self.n_layer-1):
            if self.generator_upsample[i] == 2:
    
                x = resBlock(x, self.generator_conv_filters[i], 
                             kernel_size = self.generator_conv_kernal[i],
                             strides = self.generator_conv_stride[i],
                             short_cut = self.generator_short_cut[i])
                x = UpSampling2D()(x)
                             
                    
            else:
                x = resBlock(x, self.generator_conv_filters[i], 
                             kernel_size = self.generator_conv_kernal[i],
                             strides = self.generator_conv_stride[i],
                             short_cut = self.generator_short_cut[i])
                    
        
        
        x = Conv2D(
                filters=self.generator_conv_filters[-1],
                kernel_size=self.generator_conv_kernal[-1],
                strides=1,
                padding="same"
                )(x)
        
        
        x = Activation('tanh', name="tanh")(x)
                
        
        generator_out = x
        generator = Model([generator_input, label_inp], generator_out)
        
        return generator







class Discriminator():
    def __init__(self,
                 discriminator_input,
                 label_len,
                 norm,
                 activation,
                 discriminator_upsampling,
                 discriminator_conv_filters,
                 discriminator_conv_kernal,
                 discriminator_short_cut
                 ):
        
        self.label_len = label_len
        self.discriminator_input = discriminator_input
        self.n_layer = len(discriminator_conv_filters)
        self.norm = norm
        self.activation = activation
        
        self.discriminator_upsampling = discriminator_upsampling
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernal = discriminator_conv_kernal
        self.discriminator_conv_stride = [1]*self.n_layer
        self.discriminator_short_cut = discriminator_short_cut

    
    def get_label(self, label_inp, label_len):
        
        label = Embedding(label_len, 100)(label_inp)
        x = Dense(np.prod(self.discriminator_input))(label)
        x = Reshape(self.discriminator_input)(x)
        
        return x
    
    def build_discriminator(self):
        
        
        img_label_inp = Input((1,))
        img_label = self.get_label(img_label_inp, self.label_len)
        
        discriminator_input = Input(self.discriminator_input, name="dis_input")
        
        # combine label and data
        
        x = Concatenate()([discriminator_input, img_label])
        
        x = Conv2D(
                filters=self.discriminator_conv_filters[0],
                kernel_size=self.discriminator_conv_kernal[0],
                strides=self.discriminator_conv_stride[0],
                padding="same",
                name="discriminator_conv_"+str(0),
                )(x)
      
        x = LayerNormalization()(x)
       
        x = Activation("leaky_relu")(x)
        
        
        for i in range(1, self.n_layer-1):
            
            if self.discriminator_upsampling[i] == 2:
                
                x =AveragePooling2D()(x)
                x = resBlock(x, self.discriminator_conv_filters[i],
                             kernel_size = self.discriminator_conv_kernal[i],
                             strides = self.discriminator_conv_stride[i],
                             short_cut = self.discriminator_short_cut[i],
                             norm=self.norm)
                             
                             
                            
            else:
                
                x = resBlock(x, self.discriminator_conv_filters[i],
                             kernel_size = self.discriminator_conv_kernal[i],
                             strides = self.discriminator_conv_stride[i],
                             short_cut = self.discriminator_short_cut[i],
                             norm=self.norm)
                             
                                
             
        dis_out = Conv2D(
                filters=self.discriminator_conv_filters[-1],
                kernel_size=self.discriminator_conv_kernal[-1],
                strides=2,
                padding="same",
                )(x)
        
    
        x = Activation('leaky_relu', name="leaky_relu")(dis_out)
                
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(1)(x)
        
                
                
        discriminate_out = x
        
        discriminator = Model([discriminator_input, img_label_inp], discriminate_out)
        
        return discriminator

        
        