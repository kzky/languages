from models import CNN
from datasets import DataReader
import tensorflow as tf
import numpy as np
import time
import __future__
import os

def main():
    # Setting
    n_train_data = 60000
    batch_size = 32
    epoch = 20
    n_iter = n_train_data / batch_size * epoch

    # Placeholder
    in_dim = [None, 28, 28, 1]
    out_dim = 10  # num. of class
    x = tf.placeholder(tf.float32, shape=in_dim, name="x")
    y = tf.placeholder(tf.float32, shape=[None, out_dim], name="y")

    # Model
    cnn = CNN(x, y)

    # Data
    home = os.environ["HOME"]
    train_path="{}/.chainer/dataset/pfnet/chainer/mnist/train.npz".format(home)
    test_path="{}//.chainer/dataset/pfnet/chainer/mnist/test.npz".format(home)

    data_reader = DataReader(train_path, test_path, batch_size=batch_size)

    # Optimizer
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cnn.loss)

    # Run training and test
    init_op = tf.initialize_all_variables()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Init
        sess.run(init_op)
        st = time.time()
        epoch = 0
        for i in range(n_iter):
            # Read
            x_data, y_data = data_reader.get_train_batch()

            # Train
            train_op.run(feed_dict={x: x_data, y: y_data})

            # Eval for classificatoin
            if (i+1) % (n_train_data / batch_size) == 0:
                accuracies = []
                losses = []
                data_points = []
                epoch += 1
                while True:
                    x_data, y_data = data_reader.get_test_batch()
                    if data_reader._next_position_test == 0:
                        acc_mean = 100. * \
                          np.asarray(accuracies).dot(np.asarray(data_points)) \
                          / np.sum(data_points)
                        loss_mean = 100. * \
                          np.asarray(losses).dot(np.asarray(data_points)) \
                          / np.sum(data_points)
                          
                        et = time.time()
                        msg = "Epoch={},Elapsed Time={}[s],Iter={},Loss={}Acc={}"
                        print(msg.format(epoch, et - st, i, loss_mean, acc_mean))
                        break

                    acc, loss = sess.run([cnn.accuracy, cnn.loss],
                                             feed_dict={x: x_data, y: y_data})
                    accuracies.append(acc)
                    losses.append(loss)
                    data_points.append(len(y_data))
            
if __name__ == '__main__':
    main()
