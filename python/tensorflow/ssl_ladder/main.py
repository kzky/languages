from models import SSLLadder
from datasets import DataReader, Seprator
import tensorflow as tf
import numpy as np
import time
import os
import __future__

def main():
    # Setting
    u = 100
    n_train_data = 60000
    n_l_train_data = 60000 - u
    n_u_train_data = u
    batch_size = 128
    in_dim = 784
    L = 3
    mid_dim = n_dim = 100
    out_dim = n_cls = 10
    n_epoch = 30
    n_iter = 30 * n_u_train_data / batch_size

    # Separate
    home = os.environ.get("HOME")
    fpath = os.path.join(home, ".chainer/dataset/pfnet/chainer/mnist/train.npz")
    separator = Separator(u)
    separator.separate_then_save(fpath)

    # Placeholder
    x_l = tf.placeholder(tf.float32, shape=[None, in_dim], name="x")
    y_l = tf.placeholder(tf.float32, shape=[None, out_dim], name="y")
    x_u = tf.placeholder(tf.float32, shape=[None, in_dim], name="x")

    
    # Model
    ssl_ladder = SSLLadder(x_l, y, x_u, L, n_dim, n_cls, phase_train)

    # Data
    home = os.environ["HOME"]
    train_path = \
        "{}/.chainer/dataset/pfnet/chainer/mnist/l_train.npz".format(home)
    test_path = \
        "{}/.chainer/dataset/pfnet/chainer/mnist/u_test.npz".format(home)

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
            x_l_data, y_l_data = data_reader.get_l_train_batch()
            x_u_data, y_u_data = data_reader.get_u_train_batch()

            # Train
            train_step.run(feed_dict={x_l: x_l_data, y_l: y_l_data, x_u: x_u_data})
             
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
                                         feed_dict={
                                             x_l: x_l_data,
                                             y_l: y_l_data,
                                             x_u: x_u_data,
                                             phase_train: False})
                    accuracies.append(acc)
                    losses.append(loss)
                    data_points.append(len(y_l))

if __name__ == '__main__':
    main()
