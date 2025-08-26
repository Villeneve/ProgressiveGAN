import keras.layers as lay
import keras
import numpy as np

class Discriminator(keras.Model):
    def __init__(self, resolution=4.,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_convs = int(np.log2(resolution))
        self.dense = lay.Dense(1,activation='sigmoid',name='classification_head')
        self.flatten = lay.Flatten()
        self.convs = []
        self.downsampling = []
        self.fromRGB = []
        for i in range(1+self.n_convs-2):
            for ii in range(2):
                self.convs.append(
                    lay.Conv2D(
                        filters=256,
                        kernel_size=(3,3),
                        strides=(1,1),
                        padding='same',
                        activation='leaky_relu',
                        name=f'block{i+1}_conv{ii+1}'
                    )
                )
            self.downsampling.append(
                lay.AveragePooling2D((2,2),name=f'block{i+1}_dsample1')
            )
            self.fromRGB.append(
                lay.Conv2D(
                    filters=256,
                    kernel_size=(3,3),
                    strides=(1,1),
                    padding='same',
                    activation='leaky_relu',
                    name=f'block{i+1}_fromRGB'
                )
            )
    
    def call(self,inputs,resolution=4.,fade=0.):
        resolution=np.log2(resolution)

        x = inputs
        for i in range(resolution-1):
            if fade <= 0:
                x = self.fromRGB[-1](x)
                x = self.convs[-1](x)
                x = self.convs[-2](x)
            elif
            x = self.flatten(x)
            x = self.dense(x)

        return x


