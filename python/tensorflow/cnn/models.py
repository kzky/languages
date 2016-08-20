import tensorflow as tf
import numpy as np

class CNN(object):
    """
    Attributes
    ---------------
    x: tf.placeholder
    y: tf.placeholder
    phase_train: tf.placeholder of bool used in BN
    pred: pred op
    loss: loss op, or objective function op
    accuracy: accuracy op
    """

    def __init__(self, x, y, phase_train):
        """
        Parameters
        -----------------
        x: tf.placeholder of sample
        y: tf.placeholder of label
        phase_train: tf.placeholder of bool used in BN
        """
        self._x = x
        self._y = y
        self._phase_train = phase_train
        
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
        W = tf.get_variable(name=w_name, shape=ksize,
                            initializer=tf.truncated_normal_initializer(ksize))
        b = tf.get_variable(name=b_name, shape=[ksize[-1]],
                            initializer=tf.truncated_normal_initializer(ksize[-1]))

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
        W = tf.get_variable(name=w_name, shape=[in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer([in_dim, out_dim]))
        b = tf.get_variable(name=b_name, shape=[out_dim],
                            initializer=tf.truncated_normal_initializer([out_dim]))
        
        x_ = tf.reshape(x, [-1, in_dim])
        linear_op = tf.matmul(x_, W)

        return linear_op

    def _batch_norm(self, x, name):

        # Determine affine or conv
        shape = x.get_shape()
        depth = shape[-1].value
        if len(shape) == 4:  # NHWC
            axes = [0, 1, 2]
        else:
            axes = [0]

        # Batch mean/var and gamma/beta
        batch_mean, batch_var = tf.nn.moments(x, axes=axes)
        beta_name = "beta-{}".format(name)
        gamma_name= "gamma-{}".format(name)
        beta = tf.get_variable(name=beta_name, shape=[depth],
                               initializer=tf.truncated_normal_initializer([depth]))
        gamma = tf.get_variable(name=gamma_name, shape=[depth],
                                initializer=tf.truncated_normal_initializer([depth]))

        # Moving average
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        # Train phase
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):  # ema op computed here
                return tf.identity(batch_mean), tf.identity(batch_var)
            
        mean, var = tf.cond(self._phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))

        x_normed = tf.nn.batch_normalization(
            x, mean, var, beta, gamma, variance_epsilon=1e-12)
        return x_normed

    def _inference(self, ):
        # 2 x Conv and 2 Maxpooling
        conv1 = self._conv_2d(self._x, name="conv1",
                                  ksize=[3, 3, 1, 64], strides=[1, 1, 1, 1])
        relu1 = tf.nn.relu(self._batch_norm(conv1, name="bn-conv1"))
        max_pool1 = self._max_pooling_2d(relu1, name="max_pool1",
                                             ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        conv2 = self._conv_2d(max_pool1, name="conv2",
                                  ksize=[3, 3, 64, 32], strides=[1, 1, 1, 1])
        relu2 = tf.nn.relu(self._batch_norm(conv2, name="bn-conv2"))
        max_pool2 = self._max_pooling_2d(relu2, name="max_pool2",
                                             ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        
        # 2 x Affine
        linear1 = self._batch_norm(
            self._linear(max_pool2, name="affine1", out_dim=50),
            name="bn-affine1")

        pred = self._batch_norm(
            self._linear(linear1, name="affine2", out_dim=10),
            name="bn-affine2")

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

        
