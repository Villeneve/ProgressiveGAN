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
    return x

def toRGB(x):
    x = lay.Conv2D(
        filters=3,
        kernel_size=(1,1),
        strides=(1,1),
        padding='same',
        activation='tanh',
    )(x)
    return x

def fade_in(from_layer,to_layer,alpha=0.):
    return (1-alpha)*from_layer + alpha*to_layer

def create_generator(input_shape=128):
    inputs = lay.Input(shape=(input_shape,))
    x = inputs

    x = lay.Dense(4*4*512,activation='leaky_relu')(inputs)
    x = lay.Reshape(target_shape=(4,4,512))(x)
    out4x4 = toRGB(x)

    x = block_conv(x,256)
    out8x8 = toRGB(x)

    x = block_conv(x,128)
    out16x16 = toRGB(x)

    x = block_conv(x,64)
    out32x32 = toRGB(x)

    return keras.Model(inputs,[out4x4,out8x8,out16x16,out32x32])

