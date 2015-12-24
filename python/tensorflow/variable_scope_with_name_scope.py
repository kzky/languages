import tensorflow as tf

# op.name changed
with tf.variable_scope("foo"):
    x = 1.0 + tf.get_variable("x", [1])
assert x.op.name == "foo/add"

# op name only affected by using name_scope
with tf.variable_scope("hoo") as hoo_scope:
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        x1 = 1.0 + v
assert v.name == "hoo/v:0"
assert x1.op.name == "hoo/bar/add"

# captured object passed to variable scope
with tf.variable_scope(hoo_scope):
    with tf.name_scope("bar"):
        v1 = tf.get_variable("v1", [1])
        x2 = 1.0 + v1
        x3 = 1.0 - v1
        print v1.name
        print x2.op.name
        print x3.op.name

# op name only affected by using name_scope
with tf.name_scope("hoo"):
    with tf.variable_scope("bar"):
        v2 = tf.get_variable("v2", [1])
        x4 = 1.0 + v2
        print v2.name
        print x4.op.name




