import keras
import keras.layers as lay
import tensorflow as tf
import numpy as np
from src.layer import Fade_in, MiniBatchSTD, PixelNorm

class Discriminator(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dsample = [lay.AvgPool2D((2,2),name=f'downsample{i}') for i in ['4_0','4_1','8_0','8_1','16_0','16_1',]]
        self.minibatchSTD = MiniBatchSTD()
        self.brain = lay.Dense(1,activation='sigmoid',name='Brain')
        self.fade_in = [Fade_in(name=f'fade_in_{i}') for i in ['4_8','8_16','16_32']]
        self.stage = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.fromRGB = [
            lay.Conv2D(
                filters=128,
                kernel_size=(3,3),
                strides=(1,1),
                padding='same',
                activation='leaky_relu',
                name=f'fromRGB{i}'
            ) for i in [4,8,16,32]
        ]

        # Camadas convolucionais
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

        self.flat = lay.Flatten(name='flat_layer')

    def build(self):
        # build single layers
        self.brain.build((None,4*4*128))
        self.flat.build((None,4,4,128))
        # build convolutional layers
        self.conv2[0].build((None,4,4,128))
        self.conv2[1].build((None,4,4,129))
        self.conv2[2].build((None,8,8,128))
        self.conv2[3].build((None,8,8,128))
        self.conv2[4].build((None,16,16,128))
        self.conv2[5].build((None,16,16,128))
        self.conv2[6].build((None,32,32,128))
        self.conv2[7].build((None,32,32,128))
        # build RGB layers
        self.fromRGB[0].build((None,4,4,3))
        self.fromRGB[1].build((None,8,8,3))
        self.fromRGB[2].build((None,16,16,3))
        self.fromRGB[3].build((None,32,32,3))
        # build fade layers

    def call(self, inputs, stage=0,*args, **kwargs):
        def forward_4x4(inputs):
            x = self.fromRGB[0](inputs) # (None,4,4,128)
            x = self.conv2[0](x)        # (None,4,4,128)
            x = self.minibatchSTD(x)
            x = self.conv2[1](x)        # (None,4,4,128)
            x = self.flat(x)            # (None,2048)
            x = self.brain(x)           # (None,1)
            return x
        
        def forward_8x8(inputs):
            # New path
            x = self.fromRGB[1](inputs) # (None,8,8,128)
            x = self.conv2[2](x)        # (None,8,8,128)
            x = self.conv2[3](x)        # (None,8,8,128)
            s8 = self.dsample[0](x)     # (None,4,4,128)
            # Old path
            x = self.dsample[1](inputs) # (None,4,4,128)
            s4 = self.fromRGB[0](x)     # (None,4,4,128)

            x = self.fade_in[0]([s4,s8])# (None,4,4,128)
            x = self.conv2[0](x)
            x = self.minibatchSTD(x)
            x = self.conv2[1](x)
            x = self.flat(x)
            x = self.brain(x)
            return x
        
        def forward_16x16(inputs):
            # New path
            x = self.fromRGB[2](inputs)
            x = self.conv2[4](x)
            x = self.conv2[5](x)
            s16 = self.dsample[2](x)
            # Old path
            x = self.dsample[3](inputs)
            s8 = self.fromRGB[1](x)

            x = self.fade_in[1]([s8,s16])
            x = self.conv2[2](x)
            x = self.conv2[3](x)
            x = self.dsample[0](x)
            x = self.conv2[0](x)
            x = self.minibatchSTD(x)
            x = self.conv2[1](x)
            x = self.flat(x)
            x = self.brain(x)
            return x
        
        def forward_32x32(inputs):
            # New path
            x = self.fromRGB[3](inputs)
            x = self.conv2[6](x)
            x = self.conv2[7](x)
            s32 = self.dsample[4](x)
            # Old path
            x = self.dsample[5](inputs)
            s16 = self.fromRGB[2](x)

            x = self.fade_in[2]([s16,s32])
            x = self.conv2[4](x)
            x = self.conv2[5](x)
            x = self.dsample[2](x)
            x = self.conv2[2](x)
            x = self.conv2[3](x)
            x = self.dsample[0](x)
            x = self.conv2[0](x)
            x = self.minibatchSTD(x)
            x = self.conv2[1](x)
            x = self.flat(x)
            x = self.brain(x)
            return x
        
        if stage == 0: return forward_4x4(inputs)
        elif stage == 1: return forward_8x8(inputs)
        elif stage == 2: return forward_16x16(inputs)
        elif stage == 3: return forward_32x32(inputs)