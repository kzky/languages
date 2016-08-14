import tensorflow as tf
import numpy as np

def inference(x, y):
    # Input/Ouput data
    x_shape = x.get_shape()
    in_dim = x_shape[1].value

    y_shape = y.get_shape()
    out_dim = y_shape[1].value

    # 2 layers MLP
    W0 = tf.Variable(tf.truncated_normal(
        shape=[in_dim, 100], mean=0.0, stddev=0.01, dtype=tf.float32))
    b0 = tf.Variable(tf.truncated_normal(
        shape=[100], mean=0.0, stddev=0.01, dtype=tf.float32))
    h0 = tf.matmul(x, W0) + b0

    W1 = tf.Variable(tf.truncated_normal(
        shape=[100, out_dim], mean=0.0, stddev=0.01, dtype=tf.float32))
    b1 = tf.Variable(tf.truncated_normal(
        shape=[out_dim], mean=0.0, stddev=0.01, dtype=tf.float32))
    h1 = tf.matmul(h0, W1) + b1

    # Softmax
    pred = tf.nn.softmax(h1)

    return pred

def compute_loss(pred, y):
    cross_entropy = -tf.reduce_sum(y * tf.log(pred), reduction_indices=[1])
    cross_entropy_over_batch = tf.reduce_mean(cross_entropy)

    return cross_entropy_over_batch

def main():

    # General Settings
    batch_size = 128
    in_dim=784
    out_dim=10
    input_name="x"
    output_name = "y"

    # Build computational graph
    x = tf.placeholder(tf.float32, shape=[None, in_dim], name=input_name)
    y = tf.placeholder(tf.float32, shape=[None, out_dim], name=output_name)
    pred_op = inference(x, y)
    loss_op = compute_loss(pred_op, y)

    with tf.Session() as sess:
        # Init trainable parameters
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        # Run one batch
        x_ = np.random.rand(batch_size, in_dim)
        y_ = np.random.rand(batch_size, out_dim)

        #loss = sess.run(loss_op, feed_dict={x: x_, y: y_})
        loss = loss_op.eval(feed_dict={x: x_, y: y_})
        print loss

if __name__ == '__main__':
    main()
    
