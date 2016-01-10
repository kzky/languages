#!/usr/bin/env python

import tensorflow as tf
import numpy as np


# Collection keys
print "### Collection keys ###"
print dir(tf.GraphKeys)
print ""

# Show values in VARIABLES collections
print "### Show values in VARIABLES collections ###"
print tf.get_collection(tf.GraphKeys.VARIABLES)
print ""

# Create Varialbes and show the colllections again
print "### Create Varialbes and show the colllections again ###"
x = tf.Variable(np.random.rand(5, 5, 5), name="x")
y = tf.Variable(np.random.rand(5, 5, 5), name="y")
c = tf.constant(np.random.rand(5, 5, 5), name="c")
init_op = tf.initialize_all_variables()

print "VARIABLES", tf.get_collection(tf.GraphKeys.VARIABLES)
print "TRAINABLE_VARIABLES", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print "TABLE_INITIALIZERS", tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
print "SUMMARIES", tf.get_collection(tf.GraphKeys.SUMMARIES)
print "QUEUE_RUNNERS", tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
print ""

# Add somthing to any collection and show that
print "### Add somthing to any collection ###"
sample = tf.get_collection("sample")
sample.append(x)
print tf.get_collection("sample")

tf.add_to_collection("sample", x)
tf.add_to_collection("sample", y)
print tf.get_collection("sample")
print ""

# Add somthing to any collection and show that with scope filter
print "### Add somthing to any collection and show that with scope filter ###"
tf.add_to_collection("sample", x)
with tf.name_scope("name_scope") as scope:
    print tf.get_collection("sample", scope)
    z = tf.Variable(np.random.rand(5, 5, 5), name="z")
    tf.add_to_collection("sample", z)
    
print len(tf.get_collection("sample"))
print len(tf.get_collection("sample", scope))

print x.name
print y.name
print z.name
