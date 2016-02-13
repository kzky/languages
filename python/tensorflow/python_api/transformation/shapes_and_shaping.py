#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# tf.shape
x = np.random.rand(5, 6, 7)
z_shape = tf.shape(x)

# tf.size
x = np.random.rand(5, 6, 7)
z_size = tf.size(x)

# tf.rank
x = np.random.rand(5, 6, 7)
z_rank = tf.rank(x)

# tf.reshape
x = np.random.rand(5, 6, 7)
z_reshape = tf.reshape(x, shape=(2, 5, 7, 3))
z_flatten = tf.reshape(x, shape=[-1])

# tf.squeeze
x = np.random.rand(1, 2, 1, 3, 1)
z_squeeze = tf.squeeze(x, squeeze_dims=[0, 4])

# tf.expand_dims
x = np.random.rand(3, 5)
z_expand_dims = tf.expand_dims(x, dim=0)

with tf.Session() as sess:

    print "tf.shape"
    print sess.run(z_shape)
    print "tf.size"
    print sess.run(z_size)
    print "tf.rank"
    print sess.run(z_rank)
    print "tf.reshape"
    print sess.run(z_reshape)
    print "tf.flatten"
    print sess.run(z_flatten)
    print "tf.squeeze"
    print sess.run(z_squeeze)
    print "tf.expand_dims"
    print sess.run(z_expand_dims)
