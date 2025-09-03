import tensorflow as tf

@tf.function
def train_step(gan,batch,bce,opt,stage=0):

    gen, dis = gan
    gen.stage.assign(stage)
    dis.stage.assign(stage)

    tf.print(f'DStage = {dis.stage.read_value()}; GStage = {gen.stage.read_value()}')

    batch_size = tf.shape(batch)[0]
    noise = tf.random.normal((batch_size*2,128))
    with tf.GradientTape() as tape:
        fake_imgs = gen(noise,training=True)
        fake_logits = dis(fake_imgs,training=True)
        g_loss = bce(tf.ones_like(fake_logits),fake_logits)
    grads = tape.gradient(g_loss,gen.trainable_variables)
    opt[0].apply_gradients(zip(grads,gen.trainable_variables))

    noise = tf.random.normal((batch_size,128))
    with tf.GradientTape() as tape:
        fake_imgs = gen(noise,training=True)
        fake_logits = dis(fake_imgs,training=True)
        true_logits = dis(batch,training=True)
        d_loss = (bce(tf.ones_like(true_logits),true_logits)+bce(tf.zeros_like(fake_logits),fake_logits))/2.
    grads = tape.gradient(d_loss,dis.trainable_variables)
    opt[1].apply_gradients(zip(grads,dis.trainable_variables))

    return g_loss, d_loss
