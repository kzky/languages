#!/usr/bin/env python
import numpy as np
import tensorflow as tf

b = tf.constant(np.random.rand(5), name="b")
x = tf.Variable(np.random.rand(10, 5), name="x")
W = tf.Variable(np.random.rand(10, 10), name="W")
h = tf.matmul(W, x) + b

c = tf.constant(np.random.rand(5), name="c")
y = tf.Variable(np.random.rand(10, 5), name="y")
V = tf.Variable(np.random.rand(10, 10), name="V")
g = tf.matmul(V, y) + c

with tf.control_dependencies([h, g]):
    #h_sum = tf.reduce_sum(h)
    #g_sum = tf.reduce_sum(g)
    # 
    #if tf.greater(h_sum, g_sum): # can not execute eval here, so that this is always True
    #    f = tf.Variable(1)
    #else:
    #    f = tf.Variable(0)

    condition = tf.greater(h, g)
    f = tf.select(condition, h, g)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    
    #ret_h = sess.run(h)
    #ret_g = sess.run(g)
    #print ret_h, ret_g

    #ret_f, ret_h_sum, ret_g_sum = sess.run([f, h_sum, g_sum])
    ret_f = sess.run(f)

    #print ret_h_sum, ret_g_sum
    print ret_f

    
    





