"""
SSL-Ladder network.

.. [1] Antti Rasmus, Mathias Berglund, Mikko Honkala, Harri Valpola, Tapani Raiko: Semi-supervised Learning with Ladder Networks. NIPS 2015: 3546-3554
.. [2] Mohammad Pezeshki, Linxi Fan, Philemon Brakel, Aaron C. Courville, Yoshua Bengio: Deconstructing the Ladder Network Architecture. ICML 2016: 2368-2376

"""

import tensorflow as tf
import numpy as np

class SSLLadder(object):
    """
    Attributes
    ---------------
    x_l: tf.placeholder
        Dimension of x 2-d whose shape is [None, 784] in case of MNIST
    y_l: tf.placeholder
    x_u: tf.placeholder
        Dimension of x 2-d whose shape is [None, 784] in case of MNIST
    phase_train: tf.placeholder of bool
        Used in BN
    lambda_lit: list of int
    std: float
    pred: tf.Tesnor
        Predictor of the network.
    corrupted_encoder: tf.Tesnor
        Corrupted Encoder of SSL Ladder Network
    clean_encoder: tf.Tesnor
        Clean Encoder of SSL Ladder Network
    decoder: tf.Tesnor
        Decoder of SSL Ladder Network
    loss: tf.Tesnor
        Loss function, or the objective function
    accuracy: tf.Tesnor
        Accuracy
    """
    
    def __init__(self, x_l, y_l, x_u, n_dims, n_cls, phase_train, lambda_list, std=0.3):
        """
        Parameters
        -----------------
        x_l: tf.placeholder
            tf.placeholder of sample
        y_l: tf.placeholder
            tf.placeholder of label
        x_u: tf.placeholder
            tf.placeholder of unlabeled sample
        n_dim: int
            Number of dimensions
        n_cls: int
            Number of classes
        phase_train: tf.placeholder of bool
            tf.placeholder of bool Used in BN

        """
        self._x_l = x_l
        self._y_l = y_l
        self._x_u = x_u
        self._L = len(n_dims) - 1
        self._n_dims = n_dims
        self._n_cls = n_cls
        self._phase_train = phase_train
        
        self.pred = None
        self.h_noise = None
        self.corrupted_encoder = None
        self.clean_encoder = None
        self.decoder = None
        self.loss = None
        self.accuracy = None

        self._lambda_list = lambda_list
        self._std = std

        # Build Graph
        self._build_graph()
        self._accuracy()
        self._add_summaries()

    def _build_graph(self, ):
        """Build the computational graph
        """
        l_loss = self._construct_ssl_ladder(self._x_l, self._y_l)
        u_loss = self._construct_ssl_ladder(self._x_u, reuse=True)
        self.loss = l_loss + u_loss

    def _add_summaries(self,):
        tf.scalar_summary("accuracy", self.accuracy)
        #tf.histogram_summary("prediction", self.pred)
        
    def _conv_2d(self, x, name, variable_scope, 
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

        with variable_scope:
            W = tf.get_variable(name=w_name, shape=ksize)
            #b = tf.get_variable(name=b_name, shape=[ksize[-1]])
              
        conv2d_op = tf.nn.conv2d(x, W, strides=strides, padding=padding)
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

    def _linear(self, x, name, out_dim, variable_scope):
        """
        Parameters
        -----------------
        x: tf.Tesnor
        name: str
            Name for the parameter.
        scope_name: str
        """
        in_dim = 1
        for dim in x.get_shape()[1:]:
            in_dim *= dim.value

        w_name = "w-{}".format(name)
        b_name = "b-{}".format(name)

        with variable_scope:
            W = tf.get_variable(name=w_name, shape=[in_dim, out_dim])
            #b = tf.get_variable(name=b_name, shape=[out_dim])
            
        x_ = tf.reshape(x, [-1, in_dim])
        linear_op = tf.matmul(x_, W)

        return linear_op

    def _denoise(self, z_noise, u, name, variable_scope):
        """Denoising function

        Denoise z_noise from the lateral connection using u from the upper layer.
        Denoising function should be linaer w.r.t. z_noise.

        Parameters
        -----------------
        z_noise: tf.Tensor
            Noisy input from the lateral connection.
        u: tf.Tensor
            Normalized input from the upper layer.
        name: str
        variable_scope: tf.variable_scope

        Returns
        -----------
        tf.Tensor
            Denoised output with denoising function.
        
        """
        shape = []
        for dim in z_noise.get_shape()[1:]:
            shape.append(dim.value)

        with variable_scope:
            a_mu_1_name = "a_mu_1-{}".format(name)
            a_mu_1 = tf.get_variable(a_mu_1_name, shape=shape, initializer=tf.zeros)
            a_mu_2_name = "a_mu_2-{}".format(name)
            a_mu_2 = tf.get_variable(a_mu_2_name, shape=shape, initializer=tf.ones)
            a_mu_3_name = "a_mu_3-{}".format(name)
            a_mu_3 = tf.get_variable(a_mu_3_name, shape=shape, initializer=tf.zeros)
            a_mu_4_name = "a_mu_4-{}".format(name)
            a_mu_4 = tf.get_variable(a_mu_4_name, shape=shape, initializer=tf.zeros)
            a_mu_5_name = "a_mu_5-{}".format(name)
            a_mu_5 = tf.get_variable(a_mu_5_name, shape=shape, initializer=tf.zeros)

            a_var_1_name = "a_var_1-{}".format(name)
            a_var_1 = tf.get_variable(a_var_1_name, shape=shape, initializer=tf.zeros)
            a_var_2_name = "a_var_2-{}".format(name)
            a_var_2 = tf.get_variable(a_var_2_name, shape=shape, initializer=tf.ones)
            a_var_3_name = "a_var_3-{}".format(name)
            a_var_3 = tf.get_variable(a_var_3_name, shape=shape, initializer=tf.zeros)
            a_var_4_name = "a_var_4-{}".format(name)
            a_var_4 = tf.get_variable(a_var_4_name, shape=shape, initializer=tf.zeros)
            a_var_5_name = "a_var_5-{}".format(name)
            a_var_5 = tf.get_variable(a_var_5_name, shape=shape, initializer=tf.zeros)

        mu = a_mu_1 * tf.nn.sigmoid(a_mu_2 * u + a_mu_3) + a_mu_4 * u + a_mu_5
        var = a_var_1 * tf.nn.sigmoid(a_var_2 * u + a_var_3) + a_var_4 * u + a_var_5

        z_recon = (z_noise - mu) * var + mu

        return z_recon
        
    def _scaling_and_bias(self, x, name, variable_scope, i):
        """Scale and bias the input in BatchNorm

        The way to scale and bias is a bit different from the standard BatchNorm.
        Basically, we compute gamma * (z - mu) / std + beta,
        but here gamma * ((z - mu) / std + beta).
        Note the normalization (z - mu) / std is already computed as `x`.
                
        Parameters
        -----------------
        x: tf.Tensor
        name: str
        variable_scope: tf.variable_scope
        """
        
        # Determine affine or conv
        shape = x.get_shape()
        depth = shape[-1].value
        if len(shape) == 4:  # NHWC
            axes = [0, 1, 2]
        else:
            axes = [0]

        beta_name = "beta-{}".format(name)
        gamma_name= "gamma-{}".format(name)

        with variable_scope:
            beta = tf.get_variable(name=beta_name, shape=[depth], initializer=tf.ones)
            gamma = tf.get_variable(name=gamma_name, shape=[depth])

        if i == self._L:
            a = gamma * (x + beta)
        else:
            a = x + beta

        return a

    def _moments(self, x):
        """Compute mean and variance.

        Compute mean and variance but return std.
        In addition, update running mean.

        Parameters
        -----------------
        x: tf.Tensor

        Returns
        -----------
        tuple of tf.Tesor
            Mean and std.
        """

        # Determine affine or conv
        shape = x.get_shape()
        depth = shape[-1].value
        if len(shape) == 4:  # NHWC
            axes = [0, 1, 2]
        else:
            axes = [0]
        
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

        return mean, tf.sqrt(var + 1e-10)

    def _batch_norm(self, x, mu, std):
        """BatchNorm op.

        This is NOT Batch Normalization, Whitening Op.

        Parameters
        -----------------
        x: tf.Tensor
        mu: tf.Tensor
        std: tf.Tensor

        """
        return (x - mu) / std
        
    def _accuracy(self, ):
        """Compute accuracy op
        """
        pred = self.pred
        y = self._y_l

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy = accuracy

    def _construct_ssl_ladder(self, x, y=None, reuse=None):
        """Construct SSL Ladder Network.

        If `y` is None, the reonstruction cost is only constructed;
        otherwise the classification loss is also constructed.

        Parameters
        -----------------
        x: tf.placeholder
            x is either labeled samples or unlabeled samples.
        y: tf.placeholder
            y is not None when x is labeled samples.
        reuse: bool
            Reuse variable for parameter tying.
        
        Returns
        -----------
        tf.Tesnor
            Loss op is returned.
        
        """
        
        mu_list = []
        std_list = []
        z_list = []
        z_noise_list = []
        z_recon_bn_list = []
        lambda_list = self._lambda_list

        # Encoder
        print("# Encoder")
        h = z = x
        h_noise = z_noise = h + tf.random_normal(tf.shape(h), stddev=1.) * self._std
        z_noise_list.append(z_noise)
        mu_list.append(0)
        std_list.append(0)
        z_list.append(z)

        for i in range(1, self._L+1):
            print("\tLayer-{}".format(i))

            # Corrupted encoder
            print("\t# Corrupted encoder")

            # Variable scope
            l_variable_scope = tf.variable_scope("enc-linear", reuse=reuse)
            sb_variable_scope = tf.variable_scope("enc-scale-bias", reuse=reuse)

            # encode
            z_pre_noise = self._linear(h_noise, "{}-th".format(i), self._n_dims[i],
                                       l_variable_scope)
            mu, std = self._moments(z_pre_noise)
            z_noise = self._batch_norm(z_pre_noise, mu, std) \
                      + tf.random_normal(tf.shape(z_pre_noise), stddev=1.0) * self._std
            if i == self._L:
                h_noise = self._scaling_and_bias(z_noise, "{}-th".format(i),
                                                 sb_variable_scope, i)
            else: 
                h_noise = tf.nn.relu(self._scaling_and_bias(z_noise, "{}-th".format(i),
                                                            sb_variable_scope, i))
            z_noise_list.append(z_noise)
            
            # Clean encoder
            print("\t# Clean encoder")

            # Variable scope
            l_variable_scope = tf.variable_scope("enc-linear", reuse=True)
            sb_variable_scope = tf.variable_scope("enc-scale-bias", reuse=True)

            # encode
            z_pre = self._linear(h, "{}-th".format(i), self._n_dims[i], l_variable_scope)
            mu, std = self._moments(z_pre)
            z = self._batch_norm(z_pre, mu, std)

            if i == self._L: 
                h = self._scaling_and_bias(z, "{}-th".format(i),
                                           sb_variable_scope, i)
            else:
                h = tf.nn.relu(self._scaling_and_bias(z, "{}-th".format(i),
                                                      sb_variable_scope, i))
            mu_list.append(mu)
            std_list.append(std)
            z_list.append(z)
            
        # Set classifier
        if y is not None: 
            self.pred = h
            self.h_noise = h_noise
        
        # Decoder
        print("# Decoder")
        for i in reversed(range(0, self._L + 1)):
            print("\tLayer-{}".format(i))
            # Variable scope
            l_variable_scope = tf.variable_scope("dec-linear", reuse=reuse)
            d_variable_scope = tf.variable_scope("dec-denoise", reuse=reuse)
            
            if i == self._L:
                mu, std = self._moments(h_noise)
                u = self._batch_norm(h_noise, mu, std)
            else:
                Vz = self._linear(z_recon, "{}-th".format(i), self._n_dims[i],
                                  l_variable_scope)
                mu, std = self._moments(Vz)
                u = self._batch_norm(Vz, mu, std)

            z_noise = z_noise_list[i]
            z_recon = self._denoise(z_noise, u, "{}-th".format(i), d_variable_scope)

            mu = mu_list[i]
            std = std_list[i]

            if mu == 0 or std == 0:  # case of the first layer
                z_recon_bn = z_recon
            else:
                z_recon_bn = self._batch_norm(z_recon, mu, std)
            z_recon_bn_list.append(z_recon_bn)
            
        # Loss for both labeled and unlabeled samples
        C = 0
        z_recon_bn_list.reverse()
        for z in z_list:
            print(z.get_shape())
        for z in z_recon_bn_list:
            print(z.get_shape())
            
        for i in range(self._L + 1):
            coeff = lambda_list[i] / self._n_dims[i]
            C +=  coeff * tf.reduce_mean(
                tf.reduce_sum(tf.square(z_list[i] - z_recon_bn_list[i]), 1))
            
        # Loss for labeled samples
        if y is not None:
            C += tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.h_noise, self._y_l))

        return C

        
