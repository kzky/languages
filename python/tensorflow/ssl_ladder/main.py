from models import SSLLadder
from datasets import DataReader
import tensorflow as tf
import numpy as np
import time
import os
import __future__

def main():
    # Setting
    n_train_data = 60000
    batch_size = 128
    in_dim = 784
    mid_dim = n_dim = 100
    out_dim = n_cls = 10
    n_epoch = 30
    n_iter = 30 * n_train_data / batch_size

    # Placeholder
    x_l = tf.placeholder(tf.float32, shape=[None, in_dim], name="x")
    x_u = tf.placeholder(tf.float32, shape=[None, in_dim], name="x")
    y = tf.placeholder(tf.float32, shape=[None, out_dim], name="y")
    
    # Model
    ssl_ladder = SSLLadder(x_l, y, x_u, L, n_dim, n_cls, phase_train)

    # Data
    home = os.environ["HOME"]
    train_path="{}/.chainer/dataset/pfnet/chainer/mnist/train.npz".format(home)
    test_path="{}//.chainer/dataset/pfnet/chainer/mnist/test.npz".format(home)

    data_reader = DataReader(train_path, test_path, batch_size=batch_size)

    # Optimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(-ssl_ladder.obj)

    # Run training and test
    init_op = tf.initialize_all_variables()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Init
        sess.run(init_op)
        st = time.time()
        epoch = 0
        for i in range(n_iter):
            # Read
            x_l, y_l, x_u = data_reader.get_train_batch()

            # Train
            train_step.run(feed_dict={x: x_l})
             
            # Eval
            if (i+1) % (n_train_l / batch_size) == 0:
                objs = []
                epoch += 1
                while True:
                    #TODO: Use Saver
                    x_l, y_l = data_reader.get_test_batch()
                    if data_reader._next_position_test == 0:
                        acc_mean = 100. * \
                          np.asarray(accuracies).dot(np.asarray(data_points)) \
                          / np.sum(data_points)
                        loss_mean = 100. * \
                          np.asarray(losses).dot(np.asarray(data_points)) \
                          / np.sum(data_points)
                          
                        et = time.time()
                        msg = "Epoch={},Elapsed Time={}[s],Iter={},Loss={},Acc={}"
                        print(msg.format(epoch, et - st, i, loss_mean, acc_mean))
                        break

                    acc, loss = sess.run([ssl_ladder.accuracy, ssl_ladder.loss],
                                             feed_dict={x: x_l, y: y_l, phase_train: False})
                    accuracies.append(acc)
                    losses.append(loss)
                    data_points.append(len(y_l))

if __name__ == '__main__':
    main()
