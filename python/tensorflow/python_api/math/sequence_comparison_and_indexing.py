#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# tf.argmin
x = np.random.rand(10, 5, 4)
z_argmin = tf.argmin(x, dimension=0)

# tf.argmax
x = np.random.rand(10, 5, 4)
z_argmax = tf.argmax(x, dimension=1)


# tf.listdiff
x = np.random.randint(0, 10, 100)
y = np.random.randint(0, 10, 10)
z_listdiff = tf.listdiff(x, y)

# tf.where
x = np.random.randint(0, 2, 10 * 5 * 4)
z = np.empty(10 * 5 * 4, dtype=np.bool)
for i, x_ in enumerate(x):
    if x_ > 0:
        z[i] = True
    else:
        z[i] = False
z = z.reshape((10, 5, 4))
z_where = tf.where(z)

# tf.unique
x = np.random.randint(0, 10, 100)
z_unique = tf.unique(x)

# tf.edit_distance


# tf.invert_permutation
x = np.arange(0, 10)
np.random.shuffle(x)
z_invert_permutation = tf.invert_permutation(x)

with tf.Session() as sess:
    
    print "tf.argmin"
    print sess.run(z_argmin)
    print "z_argmax"
    print sess.run(z_argmax)
    print "tf.listdiff"
    print sess.run(z_listdiff)
    print "tf.where"
    print sess.run(z_where)
    print "tf.unique"
    print sess.run(z_unique)
    #print "tf.edit_distance"
    #print sess.run(z_edit_distance)
    print "tf.invert_permutation"
    print sess.run(z_invert_permutation)
