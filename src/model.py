import keras
import keras.layers as lay
import tensorflow as tf
from src.layer import Fade_in

class Generator(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Camadas para RGB;
        self.toRGB = [
            lay.Conv2D(
                filters=3,
                kernel_size=(1,1),
                strides=(1,1),
                padding='same',
                activation='tanh',name=f'toRGB{i}'
            ) for i in [4,8,16,32]
        ]

        # Camadas de refinamento de upsampling;
        self.conv2 = []
        for i in [4,8,16,32]:
            self.conv2.append(
                lay.Conv2D(
                filters=128,
                kernel_size=(3,3),
                strides=(1,1),
                padding='same',
                activation='leaky_relu',
                name=f'block{i}_conv0'
                )
            )
            self.conv2.append(
                lay.Conv2D(
                filters=128,
                kernel_size=(3,3),
                strides=(1,1),
                padding='same',
                activation='leaky_relu',
                name=f'block{i}_conv1'
                )
            )
        
        # Achatamento;
        self.flat = lay.Flatten(name='flat_layer')

        # Upsampling;
        self.usample = []
        for i in [8,16,32]:
            self.usample.append(
                lay.UpSampling2D(size=(2,2),interpolation="bilinear",name=f'upsample{i}')
            )
        
        # Camadas de Fade in;
        self.fade_in = []
        for i in ['4_8','8_16','16_32']:
            self.fade_in.append(
                Fade_in(name=f'fade_in({i})')
            )
        
        # Cérebro do gerador
        self.brain = lay.Dense(4*4*512,activation='leaky_relu',name='Brain')
        self.reshape = lay.Reshape((4,4,512),name='Reshape')

    def call(self, inputs, *args, **kwargs):
        
        # Forward pass para 4x4;
        x = self.brain(inputs)
        x = self.reshape(x)
        x = self.conv2[0](x)
        x = self.conv2[1](x)
        s4 = self.toRGB[0](x)

        up8 = self.usample[0](x)
        x = self.conv2[2](up8)
        x = self.conv2[3](x)
        x8 = self.toRGB[1](x)
        s8 = self.fade_in[0]([self.toRGB[0](up8),x8])

        up16 = self.usample[1](x)
        x = self.conv2[4](up16)
        x = self.conv2[5](x)
        x16 = self.toRGB[2](x)
        s16 = self.fade_in[1]([self.toRGB[1](up16),x16])

        up32 = self.usample[2](x)
        x = self.conv2[6](up32)
        x = self.conv2[7](x)
        x32 = self.toRGB[3](x)
        s32 = self.fade_in[2]([self.toRGB[2](up32),x32])
        
        return [s4,s8,s16,s32]
