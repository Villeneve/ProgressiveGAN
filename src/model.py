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