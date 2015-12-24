import tensorflow as tf
import numpy as np

v1 = tf.Variable(np.random.rand(10, 5), name="v1")
v2 = tf.Variable(np.random.rand(5, 3), name="v2")
prod = tf.matmul(v1, v2)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "/tmp/model.ckpt")
    print "Model restored."

    result = sess.run(prod)
    print result
    
