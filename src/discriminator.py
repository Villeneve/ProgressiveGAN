import keras.layers as lay
import keras
import numpy as np

class Discriminator(keras.Model):
    def __init__(self, resolution=4.,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_convs = int(np.log2(resolution))
        self.dense = lay.Dense(1,activation='sigmoid',name='classifier_head')
        self.convs = [lay.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',activation='leaky_relu') for i in range(1)]