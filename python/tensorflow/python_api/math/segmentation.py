#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# tf.segment_sum
x = np.random.rand(10, 4, 3)
segment_ids = [0, 0, 0, 1, 2, 2, 3, 4, 5, 5]
z_segment_sum = tf.segment_sum(x, segment_ids)


# tf.segment_prod
x = np.random.rand(10, 4, 3)
segment_ids = [0, 0, 0, 1, 2, 2, 3, 4, 5, 5]
z_segment_prod = tf.segment_prod(x, segment_ids)

# tf.segment_min
x = np.random.rand(10, 4, 3)
segment_ids = [0, 0, 0, 1, 2, 2, 3, 4, 5, 5]
z_segment_min = tf.segment_min(x, segment_ids)


# tf.segment_max
x = np.random.rand(10, 4, 3)
segment_ids = [0, 0, 0, 1, 2, 2, 3, 4, 5, 5]
z_segment_max = tf.segment_max(x, segment_ids)


# tf.segment_mean
x = np.random.rand(10, 4, 3)
segment_ids = [0, 0, 0, 1, 2, 2, 3, 4, 5, 5]
z_segment_mean = tf.segment_mean(x, segment_ids)


# tf.unsorted_segment_sum
x = np.random.rand(10, 4, 3)
segment_ids = [5, 0, 0, 2, 1, 2, 3, 4, 0, 5]
z_unsorted_segment_sum = tf.unsorted_segment_sum(x, segment_ids, num_segments=6)


# tf.sparse_segment_sum
x = np.random.rand(10, 4, 3)
indices = [0, 2, 3, 4]
segment_ids = [0, 0, 1, 1]
z_sparse_segment_sum = tf.sparse_segment_sum(x, indices, segment_ids)


# tf.sparse_segment_mean
x = np.random.rand(10, 4, 3)
indices = [0, 2, 3, 4]
segment_ids = [0, 1, 1, 1]
z_sparse_segment_mean = tf.sparse_segment_mean(x, indices, segment_ids)


with tf.Session() as sess:
    print "tf.segment_sum"
    print sess.run(z_segment_sum)
     
    print "tf.segment_prod"
    print sess.run(z_segment_prod)
     
    print "tf.segment_min"
    print sess.run(z_segment_min)
     
    print "tf.segment_max"
    print sess.run(z_segment_max)
     
    print "tf.segment_mean"
    print sess.run(z_segment_mean)
     
    print "tf.unsorted_segment_sum"
    print sess.run(z_unsorted_segment_sum)
     
    print "tf.sparse_segment_sum"
    print sess.run(z_sparse_segment_sum)
     
    print "tf.sparse_segment_mean"
    print sess.run(z_sparse_segment_mean)



