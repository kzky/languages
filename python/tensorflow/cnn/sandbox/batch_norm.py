import tensorflow as tf
import numpy as np

def main():

    # Afifne
    x = np.random.rand(64, 128).astype(np.float32)
    mu, var = tf.nn.moments(x, axes=[0])
    print("mu", mu.get_shape())
    print("var", var.get_shape())

    gamma = tf.Variable(tf.truncated_normal(shape=[128]))
    beta = tf.Variable(tf.truncated_normal(shape=[128]))
    bn = tf.nn.batch_normalization(x, mu, var, beta, gamma,
                                       variance_epsilon=1e-12)

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(bn))

    # Convolution
    x = np.random.rand(64, 32, 32, 128).astype(np.float32)
    mu, var = tf.nn.moments(x, axes=[0, 1, 2])
    print("mu", mu.get_shape())
    print("var", var.get_shape())
    gamma = tf.Variable(tf.truncated_normal(shape=[128]))
    beta = tf.Variable(tf.truncated_normal(shape=[128]))
    bn = tf.nn.batch_normalization(x, mu, var, beta, gamma,
                                       variance_epsilon=1e-12)
    
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(bn))

if __name__ == '__main__':
    main()
