#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# tf.random_normal
z_random_norm = tf.random_normal(
    shape=(10, 5, 5),
    mean=0.0,
    stddev=1.0)

# tf.truncated_normal
z_truncated_normal = tf.truncated_normal(
    shape=(10, 5, 5),
    mean=0.0,
    stddev=1.0)

# tf.random_uniform
z_random_uniform = tf.random_uniform(
    shape=(10, 5, 5),
    minval=-1,
    maxval=1)

# tf.random_shuffle
x = np.random.rand(10, 5, 5)
z_random_shuffle = tf.random_shuffle(x)

# tf.set_random_seed



with tf.Session() as sess:
    print "tf.random_normal"
    print sess.run(z_random_normal)

    print "tf.truncated_normal"
    print sess.run(z_truncated_normal)

    print "tf.random_uniform"
    print sess.run(z_random_uniform)

    print "tf.random_shuffle"
    print sess.run(z_random_shuffle)

    print "tf.set_random_seed"
    print sess.run(z_set_random_seed)
    
    
    pass
    pass

