import keras.layers as lay
import keras
import tensorflow as tf
import numpy as np

class Generator(keras.Model):
    def __init__(self,input_shape=(128,),n_convs=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.n_convs = n_convs
        self.upsampling = [lay.UpSampling2D(size=(2,2),interpolation='bicubic',name=f'upsampling_{i}') for i in range(3)]
        self.reshape = lay.Reshape(target_shape=(4,4,512))
        # self.conv_layers = [lay.Conv2D(filters=2**i,kernel_size=(3,3),strides=(1,1),padding='same',activation='leaky_relu', name=f'conv_{i}') for i in range(8,0,-1)]
        # self.dense_layer = lay.Dense(units=4*4*512,input_shape=input_shape,activation='leaky_relu',name='brain')
        # self.toRGB = [lay.Conv2D(filters=3,kernel_size=(1,1),strides=(1,1),padding='same',activation='tanh',name=f'toRGB_{i}') for i in range(4)]


    def build(self,input_shape):
        super().build(input_shape)
        self.conv_layers = []
        for i in range(self.n_convs):
            for _ in range(2):
                self.conv_layers.append(
                    lay.Conv2D(
                        filters=2**i,
                        kernel_size=(3,3),
                        strides=(1,1),
                        padding='same',
                        activation='leaky_relu',
                    )
                )
        self.dense_layer = lay.Dense(units=4*4*512,input_shape=input_shape,activation='leaky_relu',name='brain')
        self.toRGB = [lay.Conv2D(filters=3,kernel_size=(1,1),strides=(1,1),padding='same',activation='tanh',name=f'toRGB_{i}') for i in range(4)]

    def call(self,inputs,resolution=4.,fade=0.):
        resolution=int(np.log2(resolution)-2)
        x = self.dense_layer(inputs)
        x = self.reshape(x)
        if resolution == 0:
            if fade <= 0.:
                self.toRGB[resolution](x)
            elif fade > 0. and fade < 1.:
                x = self.upsampling[resolution]
                out1 = self.toRGB[resolution](x)
                out2 = self.conv_layers[2*resolution](x)
                out2 = self.conv_layers[2*resolution+1](out2)
                
            else:

        return x
    
    def predict_on_batch(self,inputs,resolution):
        return self(inputs=inputs,resolution=resolution)
