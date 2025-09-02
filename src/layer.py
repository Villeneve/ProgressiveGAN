import keras
import keras.layers as lay
import tensorflow as tf

class Fade_in(lay.Layer):
    def __init__(self, *, activity_regularizer=None, trainable=True, dtype=None, autocast=True, name=None, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, trainable=trainable, dtype=dtype, autocast=autocast, name=name, **kwargs)
        self.alpha = tf.Variable(0.0,trainable=False)

    def call(self,inputs,*args, **kwargs):
        old_path, new_path = inputs
        output = self.alpha * new_path + (1.-self.alpha)*old_path
        return output