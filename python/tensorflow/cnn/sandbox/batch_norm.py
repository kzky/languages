import tensorflow as tf
import numpy as np

def main():

    # Afifne
    x = np.random.rand(64, 128)
    mu, var = tf.nn.moments(x, axes=[0])
    print "mu", mu.get_shape()
    print "var", var.get_shape()

    # Convolution
    x = np.random.rand(64, 32, 32, 128)
    mu, var = tf.nn.moments(x, axes=[0, 1, 2])
    print "mu", mu.get_shape()
    print "var", var.get_shape()
    

if __name__ == '__main__':
    main()
