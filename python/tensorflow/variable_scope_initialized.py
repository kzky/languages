import tensorflow as tf


with tf.Session() as sess:


    with tf.variable_scope("foo", initializer=tf.constant_initializer(0.4)):
        v = tf.get_variable("v", [1])
        v.initializer.run()
        assert v.eval() == 0.4  # Default initializer as set above.

        w = tf.get_variable("w", [1], initializer=tf.constant_initializer(0.3))
        w.initializer.run()
        assert w.eval() == 0.3  # Specific initializer overrides the default.
     
        with tf.variable_scope("bar"):
            v = tf.get_variable("v", [1])
            v.initializer.run()
            assert v.eval() == 0.4  # Inherited default initializer.
     
        with tf.variable_scope("baz", initializer=tf.constant_initializer(0.2)):
            v = tf.get_variable("v", [1])
            v.initializer.run()
            assert v.eval() == 0.2  # Changed default initializer.

sess.close()
