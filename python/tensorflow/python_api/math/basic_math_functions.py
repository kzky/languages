#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# tf.add_n
inputs = [np.random.rand(5, 3)] * 10
z_add_n = tf.add_n(inputs)

# tf.abs
x = np.random.uniform(-1, 1, (5, 3))
z_abs = tf.abs(x)

# tf.neg
x = np.random.uniform(-1, 1, (5, 3))
z_neg = tf.neg(x)

# tf.sign
x = np.random.uniform(-1, 1, (5, 3))
z_sign = tf.sign(x)

# tf.inv
x = np.random.rand(5, 3)
z_inv = tf.inv(x)

# tf.square
x = np.random.rand(5, 3)
z_square = tf.square(x)

# tf.round
x = np.random.rand(5, 3)
z_round = tf.round(x)

# tf.sqrt
x = np.random.rand(5, 3)
z_sqrt = tf.sqrt(x)

# tf.rsqrt
x = np.random.rand(5, 3)
z_rsqrt = tf.rsqrt(x)

# tf.pow
x = np.random.rand(5, 3)
z_pow = tf.pow(x, 2)

# tf.exp
x = np.random.rand(5, 3)
z_exp = tf.exp(x)

# tf.log
x = np.random.rand(5, 3)
z_log = tf.log(x)

# tf.ceil
x = np.random.rand(5, 3)
z_ceil = tf.ceil(x)

# tf.floor
x = np.random.rand(5, 3)
z_floor = tf.floor(x)

# tf.maximum
x = np.random.rand(5, 3)
y = np.random.rand(5, 3)
z_maximum = tf.maximum(x, y)

# tf.minimum
x = np.random.rand(5, 3)
y = np.random.rand(5, 3)
z_minimum = tf.minimum(x, y)

# tf.cos
x = np.random.rand(5, 3)
z_cos = tf.cos(x)

# tf.sin
x = np.random.rand(5, 3)
z_sin = tf.sin(x)

with tf.Session() as sess:

    print "tf.add_n"
    print sess.run(z_add_n)

    print "tf.abs"
    print sess.run(z_abs)

    print "tf.neg"
    print sess.run(z_neg)

    print "tf.sign"
    print sess.run(z_sign)

    print "tf.inv"
    print sess.run(z_inv)

    print "tf.square"
    print sess.run(z_square)

    print "tf.round"
    print sess.run(z_round)

    print "tf.sqrt"
    print sess.run(z_sqrt)

    print "tf.rsqrt"
    print sess.run(z_rsqrt)

    print "tf.pow"
    print sess.run(z_pow)

    print "tf.exp"
    print sess.run(z_exp)

    print "tf.log"
    print sess.run(z_log)

    print "tf.ceil"
    print sess.run(z_ceil)

    print "tf.floor"
    print sess.run(z_floor)

    print "tf.maximum"
    print sess.run(z_maximum)

    print "tf.minimum"
    print sess.run(z_minimum)

    print "tf.cos"
    print sess.run(z_cos)

    print "tf.sin"
    print sess.run(z_sin)

