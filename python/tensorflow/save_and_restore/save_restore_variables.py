#!/usr/bin/env python

import tensorflow as tf
import datetime

def main():
    # Varialbes
    initial_v1 = tf.truncated_normal((10, 5), mean=0, stddev=0.1)
    v1 = tf.Variable(initial_v1)

    initial_v2 = tf.truncated_normal((5, 3), mean=0, stddev=0.1)
    v2 = tf.Variable(initial_v2)

    # Ops
    prod = tf.matmul(v1, v2)
    init_ops = tf.initialize_all_variables()
    
    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Init variables
        sess.run(init_ops)

        # Compute
        result0 = sess.run(prod)

        print result0

        # Save variables
        now = datetime.datetime.now()
        save_path = "/tmp/{}-{}.ckpt".format(__file__, now)
        saver.save(sess, save_path)
        print "Model saved in {}".format(save_path)

        # Restore variables (init_op is not necessary)
        saver.restore(sess, save_path)
        result1 = sess.run(prod)

        print result1

        print result0 == result1
        
if __name__ == '__main__':
    main()
