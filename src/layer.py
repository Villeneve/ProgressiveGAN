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
    
class MiniBatchSTD(lay.Layer):
    def __init__(self, *, activity_regularizer=None, trainable=True, dtype=None, autocast=True, name=None, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, trainable=trainable, dtype=dtype, autocast=autocast, name=name, **kwargs)
    
    def call(self, inputs, *args, **kwargs):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        H = shape[1]
        W = shape[2]
        C = shape[3]
        std = tf.math.reduce_std(inputs,axis=0,keepdims=True)
        mean = tf.reduce_mean(std)
        std_map = tf.fill((batch_size,H,W,1),mean)
        return tf.concat([inputs,std_map],axis=-1)
    
class PixelNorm(lay.Layer):
    def __init__(self, *, activity_regularizer=None, trainable=True, dtype=None, autocast=True, name=None, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, trainable=trainable, dtype=dtype, autocast=autocast, name=name, **kwargs)

    def call(self, inputs, *args, **kwargs):
        x = tf.square(inputs)
        x = tf.reduce_mean(x,axis=-1, keepdims=True)
        x = tf.sqrt(x+1e-8)
        return inputs/x
