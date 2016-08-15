import tensorflow as tf
import numpy as np

class CNN(object):
    """
    Attributes
    ---------------
    x: tf.placeholder
    y: tf.placeholder

    """

    def __init__(self, x, y):
        """
        Parameters
        -----------------
        x: tf.placeholder
        y: tf.placeholder
        """
        self._x = x
        self._y = y
        self.pred = None
        self.loss = None
        self.accuracy = None

        # Build Graph
        self._inference()
        self._compute_loss()
        self._accuracy()

    def _conv_2d(self, x, name,
                     ksize=[3, 3, 64, 64], strides=[1, 1, 1, 1], padding="SAME"):
        """
        Parameters
        -----------------
        x: tf.Tensor
        name: str
        ksize: list
        strides: list
        padding: str
        """
        w_name = "w-{}".format(name )
        b_name = "b-{}".format(name)
        W = tf.get_variable(name=w_name, shape=ksize)
        b = tf.get_variable(name=b_name, shape=[ksize[-1]])

        conv2d_op = tf.nn.conv2d(x, W, strides=strides, padding=padding) + b
        return conv2d_op
        
    def _max_pooling_2d(self, x, name,
                            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"):
        """
        Parameters
        -----------------
        x: tf.Tensor
        name: str
        ksize: list
        strides: list
        padding: str
        """

        max_pooling_2d_op = \
          tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)
        return max_pooling_2d_op

    def _linear(self, x, name, out_dim):
        in_dim = 1
        for dim in x.get_shape()[1:]:
            in_dim *= dim.value

        w_name = "w-{}".format(name)
        b_name = "b-{}".format(name)
        W = tf.get_variable(name=w_name, shape=[in_dim, out_dim])
        b = tf.get_variable(name=b_name, shape=[out_dim])
        
        x_ = tf.reshape(x, [-1, in_dim])
        linear_op = tf.matmul(x_, W)

        return linear_op

    def _inference(self, ):
        # 2 x Conv and 2 Maxpooling
        conv1 = tf.nn.relu(self._conv_2d(self._x, name="conv1",
                                             ksize=[3, 3, 1, 64], strides=[1, 1, 1, 1]))
        max_pool1 = self._max_pooling_2d(conv1, name="max_pool1",
                                             ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        conv2 = tf.nn.relu(self._conv_2d(max_pool1, name="conv2",
                                             ksize=[3, 3, 64, 32], strides=[1, 1, 1, 1]))
        max_pool2 = self._max_pooling_2d(conv1, name="max_pool2",
                                             ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        
        # 2 x Affine
        linear1 = self._linear(max_pool2, name="affine1", out_dim=50)
        pred = self._linear(linear1, name="affine2", out_dim=10)

        self.pred = pred
        
    def _compute_loss(self, ):
        loss = tf.nn.softmax_cross_entropy_with_logits(self.pred, self._y)
        self.loss = tf.reduce_mean(loss)

    def _accuracy(self, ):

        pred = self.pred
        y = self._y

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy = accuracy

        
