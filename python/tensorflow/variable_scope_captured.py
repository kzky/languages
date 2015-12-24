import tensorflow as tf

# VarialbeScope can be passed to tf.variable_scope
with tf.variable_scope("foo") as foo_scope:
    v = tf.get_variable("v", [1])

with tf.variable_scope(foo_scope):
    w = tf.get_variable("w", [1])

with tf.variable_scope(foo_scope, reuse=True):
    v1 = tf.get_variable("v", [1])
    w1 = tf.get_variable("w", [1])

assert v1 == v
assert w1 == w


# Jump out of the current variable scope when passing VariableScope
with tf.variable_scope("foo") as foo_scope:
    assert foo_scope.name == "foo"

with tf.variable_scope("bar"):
    with tf.variable_scope("baz") as other_scope:
        assert other_scope.name == "bar/baz"
        with tf.variable_scope(foo_scope) as foo_scope2:
            assert foo_scope2.name == "foo"  # Not changed.
