#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# tf.zeros
z_zeros = tf.zeros((5, 5, 5))

# tf.zeros_like
x = np.random.rand(5, 5, 5)
z_zeros_like = tf.zeros_like(x)

# tf.ones
z_ones = tf.ones((5, 5, 5))


# tf.ones_like
x = np.random.rand(5, 5, 5)
z_ones_like = tf.ones_like((5, 5, 5))

# tf.fill
z_fill = tf.fill((5, 5, 5), 9)

# tf.constant
x = np.random.rand(5, 5) * 10
z_constant = tf.constant(x, shape=[1, 25], dtype=tf.int16)

with tf.Session() as sess:
    print "tf.zeros"
    print sess.run(z_zeros)
    print "tf.zeros_like"
    print sess.run(z_zeros_like)
    print "tf.ones"
    print sess.run(z_ones)
    print "tf.ones_like"
    print sess.run(z_ones_like)
    print "tf.fill"
    print sess.run(z_fill)
    print "tf.constant"
    print sess.run(z_constant)
 
