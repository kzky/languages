from model import VAE
import tensorflow as tf
import numpy as np
import __future__

def main():

    # Setting
    batch_size = 128
    in_dim = 784
    mid_dim = 100

    # Placeholder
    x = tf.placeholder(tf.float32, shape=[None, in_dim], name="x")

    # Model
    vae = VAE(x)

    # Data
    dummy_data = np.round(np.random.rand(batch_size, 784))

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        
        obj = sess.run([vae.obj], feed_dict={x: dummy_data})
        print(obj)

if __name__ == '__main__':
    main()
