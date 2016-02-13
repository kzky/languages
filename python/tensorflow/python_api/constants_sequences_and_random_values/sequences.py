#!/usr/bin/env python

import tensorflow as tf

# tf.linspace
z_linspace = tf.linspace(0.0, 10.0, num=100)

# tf.range
z_range = tf.range(10, 100, delta=4)

with tf.Session() as sess:
    print "tf.linspace"
    print sess.run(z_linspace)
    print "tf.range"
    print sess.run(z_range)


