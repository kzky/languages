#!/usr/bin/env python

import tensorflow as tf

######
# Basics
######

print "# Basics"

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)

# Launch the default graph.
sess = tf.Session()

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of threes ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
result = sess.run(product)
print result
# ==> [[ 12.]]

# Close the Session when we're done.
sess.close()

#################
# With context manager
#################

print "# With context manager"
with tf.Session() as sess:
    result = sess.run([product])
    print result

#################
# Varialbes (Chainer-like)
#################
print "# Varialbes as State Example"

# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# Create an Op to add one to `var`.
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having

# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.initialize_all_variables()

# Launch the graph and run the ops.
with tf.Session() as sess:
    # Run the 'init' op
    sess.run(init_op)

    # Print the initial value of 'var'
    print sess.run(state)

    # Run the op that updates 'var' and print 'var'.
    for _ in range(3):
        sess.run(update)
        print sess.run(state)


#######
# Fetches
#######

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
    #result = sess.run([mul])
    #result = sess.run([intermed])
    result = sess.run(intermed)
    print result
    print type(result)
    result = sess.run([mul, intermed])
    print result
    print type(result[0])

###############
# Feeds (Theano-like)
###############
print "Feeds (Theano-like)"

input1 = tf.placeholder(tf.types.float32)
input2 = tf.placeholder(tf.types.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    print sess.run([output], feed_dict={input1: [7.], input2: [2.]})


