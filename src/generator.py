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
                Fade_in(name=f'fade_in{i}')
            )
        
        # Cérebro do gerador
        self.brain = lay.Dense(4*4*512,activation='leaky_relu',name='Brain')
        self.reshape = lay.Reshape((4,4,512),name='Reshape')
        self.stage = tf.Variable(0, dtype=tf.int32, trainable=False)


    def build(self):
        # build single layers
        self.brain.build((None,128))
        self.reshape.build((None,4,4,512))
        # build convolutional layers
        self.conv2[0].build((None,4,4,512))
        self.conv2[1].build((None,4,4,128))
        self.conv2[2].build((None,8,8,128))
        self.conv2[3].build((None,8,8,128))
        self.conv2[4].build((None,16,16,128))
        self.conv2[5].build((None,16,16,128))
        self.conv2[6].build((None,32,32,128))
        self.conv2[7].build((None,32,32,128))
        # build rgb layers
        self.toRGB[0].build((None,4,4,128))
        self.toRGB[1].build((None,8,8,128))
        self.toRGB[2].build((None,16,16,128))
        self.toRGB[3].build((None,32,32,128))


    def call(self, inputs, *args, **kwargs):
        
        # Forward pass para 4x4;
        def forward_4x4(inputs):
            x = self.brain(inputs)
            x = self.reshape(x)
            x = self.conv2[0](x)
            x = self.conv2[1](x)
            return self.toRGB[0](x)

        def forward_8x8(inputs):
            x = self.brain(inputs)
            x = self.reshape(x)
            x = self.conv2[0](x)
            x = self.conv2[1](x)
            x = self.usample[0](x)
            # Old path
            s4 = self.toRGB[0](x)
            # New path
            x = self.conv2[2](x)
            x = self.conv2[3](x)
            s8 = self.toRGB[1](x)
            return self.fade_in[0]([s4,s8])
        
        def forward_16x16(inputs):
            x = self.brain(inputs)
            x = self.reshape(x)
            x = self.conv2[0](x)
            x = self.conv2[1](x)
            x = self.usample[0](x)
            x = self.conv2[2](x)
            x = self.conv2[3](x)
            x = self.usample[1](x)
            # Old path
            s8 = self.toRGB[1](x)
            # New path
            x = self.conv2[4](x)
            x = self.conv2[5](x)
            s16 = self.toRGB[2](x)
            return self.fade_in[1]([s8,s16])
        
        def forward_32x32(inputs):
            x = self.brain(inputs)
            x = self.reshape(x)
            x = self.conv2[0](x)
            x = self.conv2[1](x)
            x = self.usample[0](x)
            x = self.conv2[2](x)
            x = self.conv2[3](x)
            x = self.usample[1](x)
            x = self.conv2[4](x)
            x = self.conv2[5](x)
            x = self.usample[2](x)
            # Old path
            s16 = self.toRGB[2](x)
            # New path
            x = self.conv2[6](x)
            x = self.conv2[7](x)
            s32 = self.toRGB[3](x)
            return self.fade_in[2]([s16,s32])
        
        branch_fn =[
            lambda: forward_4x4(inputs),
            lambda: forward_8x8(inputs),
            lambda: forward_16x16(inputs),
            lambda: forward_32x32(inputs)
        ]
        
        return tf.switch_case(self.stage.read_value(),branch_fn)