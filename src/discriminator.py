import keras.layers as lay
import keras

def fromRGB(x,label=1):
    x = lay.Conv2D(
        filters=
    )(x)


def create_discriminator():
    input4x4 = keras.Input((4,4,3))
    input8x8 = keras.Input((8,8,3))
    input16x16 = keras.Input((16,16,3))
    input32x32 = keras.Input((32,32,3))
