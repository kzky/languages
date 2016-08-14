from model import VAE
from datasets import DataReader
import tensorflow as tf
import numpy as np
import __future__

def main():
    # Setting
    n_train_data = 60000
    batch_size = 128
    in_dim = 784
    mid_dim = in_dim / 3
    n_iter = 10000

    # Placeholder
    x = tf.placeholder(tf.float32, shape=[None, in_dim], name="x")

    # Model
    vae = VAE(x)

    # Data
    data_reader = DataReader(batch_size=batch_size)

    # Optimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(-vae.obj)

    # Run training and test
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        # Init
        sess.run(init_op)

        for i in range(n_iter):
            # Read
            x_data, y_data = data_reader.get_train_batch()

            # Train
            train_step.run(feed_dict={x: x_data})

            # Eval
            if (i+1) % (n_train_data / batch_size) == 0:
                objs = []
                while True:
                    x_data, y_data = data_reader.get_test_batch()
                    if data_reader._next_position_test == 0:
                        obj_mean = np.mean(obj)
                        print("Iter={},Objective={}".format(i, obj))
                        break
                    obj = sess.run(vae.obj, feed_dict={x: x_data})
                    objs.append(obj)

if __name__ == '__main__':
    main()
