import tensorflow as tf

@tf.function
def train_step(gan,batch,bce,opt,stage=0):

    gen, dis = gan

    batch_size = tf.shape(batch)[0]
    noise = tf.random.normal((batch_size*2,128))
    with tf.GradientTape() as tape:
        fake_imgs = gen(noise,stage=stage,training=True)
        fake_logits = dis(fake_imgs,stage=stage,training=True)
        g_loss = bce(tf.ones_like(fake_logits),fake_logits)
    grads = tape.gradient(g_loss,gen.trainable_variables)
    opt[0].apply_gradients(zip(grads,gen.trainable_variables))

    noise = tf.random.normal((batch_size,128))
    with tf.GradientTape() as tape:
        fake_imgs = gen(noise,stage=stage,training=True)
        fake_logits = dis(fake_imgs,stage=stage,training=True)
        true_logits = dis(batch,stage=stage,training=True)
        d_loss = (bce(tf.ones_like(true_logits),true_logits)+bce(tf.zeros_like(fake_logits),fake_logits))/2.
    grads = tape.gradient(d_loss,dis.trainable_variables)
    opt[1].apply_gradients(zip(grads,dis.trainable_variables))

    return g_loss, d_loss

@tf.function
def gradient_penalty(gan, batch,stage=0):

    gen = gan[0]
    dis = gan[1]

    batch_size = tf.shape(batch)[0]

    fake_imgs = gen(tf.random.normal((batch_size,128)),stage=stage)
    alpha = tf.random.uniform((batch_size,1,1,1),0,1)
    interpolated = alpha*batch + (1-alpha)*fake_imgs

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        prediction = dis(interpolated, training=True,stage=stage)
    grads = tape.gradient(prediction,[interpolated])[0]
    grads_flat = tf.reshape(grads,(batch_size,-1))
    grads_norm = tf.norm(grads_flat,axis = -1)
    return 10*tf.reduce_mean((grads_norm - 1.0)**2)

@tf.function
def train_step_wassertein(gan,batch,bce,opt,gradient_penalty,stage=0):

    gen = gan[0]
    dis = gan[1]

    batch_size = tf.shape(batch)[0]
    noise = tf.random.normal((batch_size*2,128))
    with tf.GradientTape() as tape:
        fake_imgs = gen(noise,stage=stage,training=True)
        fake_logits = dis(fake_imgs,stage=stage,training=True)
        g_loss = -tf.reduce_mean(fake_logits)
    grads = tape.gradient(g_loss,gen.trainable_variables)
    opt[0].apply_gradients(zip(grads,gen.trainable_variables))

    noise = tf.random.normal((batch_size,128))
    with tf.GradientTape() as tape:
        fake_imgs = gen(noise,stage=stage,training=True)
        fake_logits = dis(fake_imgs,stage=stage,training=True)
        true_logits = dis(batch,stage=stage,training=True)
        d_loss = -tf.reduce_mean(true_logits - fake_logits) + gradient_penalty(gan,batch,stage)
    grads = tape.gradient(d_loss,dis.trainable_variables)
    opt[1].apply_gradients(zip(grads,dis.trainable_variables))

    return g_loss, d_loss