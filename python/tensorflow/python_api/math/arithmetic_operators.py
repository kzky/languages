#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# tf.add
x = np.random.rand(5, 3)
y = np.random.rand(5, 3)
z_add = tf.add(x, y)

# tf.sub
x = np.random.rand(5, 3)
y = np.random.rand(5, 3)
z_sub = tf.sub(x, y)

# tf.mul
x = np.random.rand(5, 3)
y = np.random.rand(5, 3)
z_mul = tf.mul(x, y)

# tf.div
x = np.random.rand(5, 3)
y = np.random.rand(5, 3)
z_div = tf.div(x, y)

# tf.truediv
## the result is always floating point
x = (np.random.rand(5, 3) * 10).astype(np.int)
y = (np.random.rand(5, 3) * 10).astype(np.int)
z_truediv = tf.truediv(x, y)

# tf.floordiv
## the result is always an integer
x = np.random.rand(5, 3)
y = np.random.rand(5, 3)
z_floordiv = tf.floordiv(x, y)

# tf.mod
x = np.random.rand(5, 3)
y = np.random.rand(5, 3)
z_mod = tf.mod(x, y)

with tf.Session() as sess:
    print "z_add"
    print sess.run(z_add)

    print "z_sub"
    print sess.run(z_sub)

    print "z_mul"
    print sess.run(z_mul)

    print "z_div"
    print sess.run(z_div)

    print "z_truediv"
    print sess.run(z_truediv)

    print "z_floordiv"
    print sess.run(z_floordiv)

    print "z_mod"
    print sess.run(z_mod)
