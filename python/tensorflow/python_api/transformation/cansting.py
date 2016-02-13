#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# tf.string_to_number
x = np.empty(10, dtype="|S10")
for i, e in enumerate(x):
    x[i] = "string_{}".format(i)
z_string_to_number = tf.string_to_number(x, out_type=tf.int32)

# tf.to_double
x = np.random.rand(3, 5)
z_to_double = tf.to_double(x)

# tf.to_float
x = np.random.rand(3, 5)
z_to_float = tf.to_float(x)

# tf.to_bfloat16
x = np.random.rand(3, 5).astype(np.float32)
z_to_bfloat16 = tf.to_bfloat16(x)

# tf.to_int32
x = np.random.rand(3, 5) * 10
z_to_int32 = tf.to_int32(x)

# tf.to_int64
x = np.random.rand(3, 5) * 10
z_to_int64 = tf.to_int64(x)

# tf.cast
x = np.random.rand(3, 5) * 10
z_cast = tf.cast(x, dtype=tf.int16)

with tf.Session() as sess:

    #print "tf.string_to_number"
    #print sess.run(z_string_to_number)
    
    print "tf.to_double"
    print sess.run(z_to_double)
    
    print "tf.to_float"
    print sess.run(z_to_float)
    
    print "tf.to_bfloat16"
    print sess.run(z_to_bfloat16)
    
    print "tf.to_int32"
    print sess.run(z_to_int32)
    
    print "tf.to_int64"
    print sess.run(z_to_int64)
    
    print "tf.cast"
    print sess.run(z_cast)

