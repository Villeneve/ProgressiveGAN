import keras.layers as lay

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

def create_generator(input_shape=128):
    inputs = lay.Input(shape=(128,))
    x = 
    x = lay.Dense(4*4*512,activation='leaky_relu')(inputs)
    out1 = lay.Reshape(target_shape=(4,4,512))(x)

