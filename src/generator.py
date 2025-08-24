import keras.layers as lay
import keras

def block_conv(x,n_filters):
    x = lay.Conv2D(
        filters=n_filters,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='leaky_relu'
    )(x)
    x = lay.Conv2D(
        filters=n_filters,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='leaky_relu'
    )(x)
    x = lay.UpSampling2D(
        size=(2,2)
    )(x)

def toRGB(x):
    x = lay.Conv2D(
        filters=3,
        kernel_size=(1,1),
        strides=(1,1),
        padding='same',
        activation='tanh',
        name='toRGB'
    )

def fade_in(from_layer,to_layer,alpha=0.):
    return (1-alpha)*from_layer + alpha*to_layer

def create_generator(input_shape=128):
    inputs = lay.Input(shape=(input_shape,))
    x = inputs
    x = lay.Dense(4*4*512,activation='leaky_relu')(inputs)
    out4x4 = lay.Reshape(target_shape=(4,4,512))(x)

    out8x8 = block_conv(out4x4,256)
    out16x16 = block_conv(out8x8,128)
    out32x32 = block_conv(out16x16,64)
    return keras.Model(inputs,[out4x4,out8x8,out16x16,out32x32])

