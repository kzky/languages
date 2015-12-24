import tensorflow as tf
import numpy as np

# Create some variables.

v1 = tf.Variable(np.random.rand(10, 5), name="v1")
v2 = tf.Variable(np.random.rand(5, 3), name="v2")
prod = tf.matmul(v1, v2)

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
    sess.run(init_op)

    # Do some work with the model.
    result = sess.run(prod)
    print result

    # Save the variables to disk.
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print "Model saved in file: ", save_path

