from models import SSLLadder
from datasets import DataReader, Separator
import tensorflow as tf
import numpy as np
import time
import os
import __future__

def main():
    # Setting
    l = 100
    n_train_data = 60000
    n_l_train_data = l
    n_u_train_data = n_train_data - l
    batch_size = 128
    in_dim = 784
    L = 3
    mid_dim = n_dim = 500
    out_dim = n_cls = 10
    n_epoch = 30
    n_iter = 30 * n_u_train_data / batch_size

    # Separate
    home = os.environ.get("HOME")
    fpath = os.path.join(home, ".chainer/dataset/pfnet/chainer/mnist/train.npz")
    separator = Separator(l)
    separator.separate_then_save(fpath)

    # Placeholder
    x_l = tf.placeholder(tf.float32, shape=[None, in_dim], name="x_l")
    y_l = tf.placeholder(tf.float32, shape=[None, out_dim], name="y_l")
    x_u = tf.placeholder(tf.float32, shape=[None, in_dim], name="x_u")
    phase_train = tf.placeholder(tf.bool, name="phase_train")
    
    # Model
    ssl_ladder = SSLLadder(x_l, y_l, x_u, L, n_dim, n_cls, phase_train)

    # Data
    home = os.environ["HOME"]
    l_train_path = \
        "{}/.chainer/dataset/pfnet/chainer/mnist/l_train.npz".format(home)
    u_train_path = \
        "{}/.chainer/dataset/pfnet/chainer/mnist/u_train.npz".format(home)
    test_path = \
        "{}/.chainer/dataset/pfnet/chainer/mnist/test.npz".format(home)

    data_reader = DataReader(l_train_path, u_train_path,
                             test_path, batch_size=batch_size)

    # Optimizer
    train_step = tf.train.AdamOptimizer(1e-4).minimize(ssl_ladder.loss)

    # Run training and test
    init_op = tf.initialize_all_variables()
    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Init
        writer = tf.train.SummaryWriter("./logs", sess.graph)
        sess.run(init_op)
        st = time.time()
        epoch = 0
        for i in range(n_iter):
            # Read
            x_l_data, y_l_data = data_reader.get_l_train_batch()
            x_u_data, y_u_data = data_reader.get_u_train_batch()

            # Train
            train_step.run(feed_dict={x_l: x_l_data, y_l: y_l_data,
                                      x_u: x_u_data, phase_train: True})
             
            # Eval
            if (i+1) % (n_u_train_data / batch_size) == 0:
                epoch += 1
                x_l_data, y_l_data = data_reader.get_test_batch()
                summary, acc = sess.run([merged, ssl_ladder.accuracy],
                               feed_dict={
                                   x_l: x_l_data,
                                   y_l: y_l_data,
                                   x_u: x_u_data,
                                   phase_train: False})
                et = time.time()
                writer.add_summary(summary)
                saver.save(sess, "./model.ckpt")
                msg = "Epoch={},Elapsed Cum Time={}[s],Iter={},Acc={}%"
                print(msg.format(epoch, et - st, i, acc * 100))

if __name__ == '__main__':
    main()
