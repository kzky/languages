import tensorflow as tf
import numpy as np

class SSLLadder(object):
    """
    Attributes
    ---------------
    x: tf.placeholder
    y: tf.placeholder
    phase_train: tf.placeholder of bool used in BN
    pred: pred op
    corrupted_encoder:
    clean_encoder: 
    decoder: 
    loss: loss op, or objective function op
    accuracy: accuracy op
    """

    def __init__(self, x_l, y, x_u, L, n_cls, phase_train):
        """
        Parameters
        -----------------
        x_l: tf.placeholder of sample
        y: tf.placeholder of label
        x_u: tf.placeholder of unlabeled sample
        L: number of layers
        n_cls: number of classes
        phase_train: tf.placeholder of bool used in BN
        """
        self._x = x
        self._y = y
        self._x_u = x_u
        self._L = L
        self._n_cls = n_cls
        self._phase_train = phase_train
        
        self.pred = None
        self.corrupted_encoder = None
        self.clean_encoder = None
        self.decoder = None
        self.loss = None
        self.accuracy = None

        # Build Graph
        self._construct_ssl_ladder()

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

    def _construct_ssl_ladder(self, ):
        
        mu_list = []
        std_list = []
        z_list = []
        z_noise_list = []
        z_recon_list = []
        lambda_list = [1] * self._L

        #TODO: move tf.sqrt to _moments
        #TODO: make batch_norm function
        #TODO: code loss
        #TODO: code classifier
        #TODO: code accuracy
        #TODO: separate labeled and unlabeled sample
        
        # Encoder
        h = self._x
        h_noise = h + tf.truncated_normal(h.get_shape())
        for i in range(self._L):
            # clean encoder
            z_pre = self._linear(h, name="{}-th".format(i), 100)

            mu, var = self._moments(z_pre)
            mu_list.append(mu)
            std = tf.sqrt(var))
            std_list.append(std)

            z = (z_pre - mu) / std
            z.append(z)
            h = tf.nn.tanh(self._scaling_and_bias(z, name="{}-th".format(i)))

            # corrupted encoder
            Wh = self_linear(h_noise, name="{}-th".format(i), 100)

            mu, var = self._moments(Wh)
            std = tf.sqrt(var))

            z_noize = (Wh - mu) / std + tf.truncated_normal(Wh.get_shape())
            z_noise_list.append(z_noize)
            h_noise = tf.nn.tanh(self._scaling_and_bias(z_noise, name="{}-th".format(i)))
            
        # Decoder
        for i in range(self._L).reverse():
            if i == self._L - 1:
                mu, var = self._moments(h_noise)
                std = tf.sqrt(var)
                u = (h_noise - mu) / std
            else:
                Vz_recon = self._linear(z_recon, name="{}-th".format(i), 100)
                mu, var = self._moments(Vz_recon)
                std = tf.sqrt(var)
                u = (Vz_recon - mu) / std

            z_noise = z_noise_list[i]
            z_recon = self._g(z_noize, u)

            mu = mu_list[i]
            var = var_list[i]
            std = tf.sqrt(var)
            z_recon_bn = (z_recon - mu) / std

        # Loss

        # Acc
        

    def _g(self, z_noise, u):
        pass
            

    def _scaling_and_bias(self, x, name):
        # Determine affine or conv
        shape = x.get_shape()
        depth = shape[-1].value
        if len(shape) == 4:  # NHWC
            axes = [0, 1, 2]
        else:
            axes = [0]

        beta_name = "beta-{}".format(name)
        gamma_name= "gamma-{}".format(name)
        beta = tf.get_variable(name=beta_name, shape=[depth])
        gamma = tf.get_variable(name=gamma_name, shape=[depth])

        return gamma * (x - beta)

    def _moments(self, x):
        # Batch mean/var and gamma/beta
        batch_mean, batch_var = tf.nn.moments(x, axes=axes)

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

        return mean, var

        
    def _compute_loss(self, ):
        loss = tf.nn.softmax_cross_entropy_with_logits(self.pred, self._y)
        self.loss = tf.reduce_mean(loss)

    def _accuracy(self, ):

        pred = self.pred
        y = self._y

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy = accuracy

        
