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
    x: tf.placeholder
        dimension of x 2-d whose shape is [None, 784] in case of MNIST
    y: tf.placeholder
    phase_train: tf.placeholder of bool
        used in BN
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
    
    def __init__(self, x_l, y, x_u, L, n_dim, n_cls, phase_train):
        """
        Parameters
        -----------------
        x_l: tf.placeholder
            tf.placeholder of sample
        y: tf.placeholder
            tf.placeholder of label
        x_u: tf.placeholder
            tf.placeholder of unlabeled sample
        L: int
            Number of layers
        n_dim: int
            Number of dimensions
        n_cls: int
            Number of classes
        phase_train: tf.placeholder of bool
            tf.placeholder of bool Used in BN

        """
        self._x = x
        self._y = y
        self._x_u = x_u
        self._L = L
        self._n_dim = _n_dim
        self._n_cls = n_cls
        self._phase_train = phase_train
        
        self.pred = None
        self.corrupted_encoder = None
        self.clean_encoder = None
        self.decoder = None
        self.loss = None
        self.accuracy = None

        # Build Graph
        self._build_graph()

    def _build_graph(self, ):
        """Build the computational graph
        """
        self.loss = self._construct_ssl_ladder(self._x, self._y) \
          + self._construct_ssl_ladder(self._x_u)

    def _get_variable_by_name(self, name):
        """Get the variable by specified name

        This is used for the parameter tying
        """
        variables = tf.get_collection(tf.GraphKeys.VARIABLES)
        for v in varialbes:
            if v.name == name:
                return v

        return None
        
    def _conv_2d(self, x, name,
                     ksize=[3, 3, 64, 64], strides=[1, 1, 1, 1], padding="SAME",
                     scope_name="conv_2d"):
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
        #b_name = "b-{}".format(name)

        with tf.variable_scope(scope_name):
            v = self._get_variable_by_name(w_name)
            W = tf.get_variable(name=w_name, shape=ksize) \
              if v is None else v
            v = self._get_variable_by_name(b_name)
            #b = tf.get_variable(name=b_name, shape=[ksize[-1]]) \
            #  if v is None else v
              
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

    def _linear(self, x, name, out_dim, scope_name="linear"):
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
        #b_name = "b-{}".format(name)

        with tf.variable_scope(scope_name):
            v = self._get_variable_by_name(w_name)
            if v is None:
            W = tf.get_variable(name=w_name, shape=[in_dim, out_dim]) \
              if v is None else v
            #v = tf.get_variable(name=b_name, shape=[out_dim])
            #b = self._get_variable_by_name(w_name) \
            #  if v is None else v
            
        x_ = tf.reshape(x, [-1, in_dim])
        linear_op = tf.matmul(x_, W)

        return linear_op

    def _denoise(self, z_noise, u, name, scope_name="denoise"):
        """Denoising function

        Denoise z_noize from the lateral connection using u from the upper layer.
        Denoising function should be linaer w.r.t. z_noize.

        Parameters
        -----------------
        z_noise: tf.Tensor
            Noisy input from the lateral connection.
        u: tf.Tensor
            Normalized input from the upper layer.
        name: str
        scope_name: str

        Returns
        -----------
        tf.Tensor
            Denoised output with denoising function.
        
        """
        shape = []
        for dim in z_noize.get_shape()[1:]:
            shape.append(dim.value)

        with tf.variable_scope(scope_name):
            a_mu_1_name = "a_mu_1-{}".format(name)
            a_mu_1 = tf.get_variable(a_mu_1_name, shape=shape)
            a_mu_2_name = "a_mu_2-{}".format(name)
            a_mu_2 = tf.get_variable(a_mu_2_name, shape=shape)
            a_mu_3_name = "a_mu_3-{}".format(name)
            a_mu_3 = tf.get_variable(a_mu_3_name, shape=shape)
            a_mu_4_name = "a_mu_4-{}".format(name)
            a_mu_4 = tf.get_variable(a_mu_4_name, shape=shape)
            a_mu_5_name = "a_mu_5-{}".format(name)
            a_mu_5 = tf.get_variable(a_mu_5_name, shape=shape)

            a_var_1_name = "a_var_1-{}".format(name)
            a_var_1 = tf.get_variable(a_var_1_name, shape=shape)
            a_var_2_name = "a_var_2-{}".format(name)
            a_var_2 = tf.get_variable(a_var_2_name, shape=shape)
            a_var_3_name = "a_var_3-{}".format(name)
            a_var_3 = tf.get_variable(a_var_3_name, shape=shape)
            a_var_4_name = "a_var_4-{}".format(name)
            a_var_4 = tf.get_variable(a_var_4_name, shape=shape)
            a_var_5_name = "a_var_5-{}".format(name)
            a_var_5 = tf.get_variable(a_var_5_name, shape=shape)

        mu = a_mu_1 * tf.nn.sigmoid(a_mu_2 * u + a_mu_3) + a_mu_4 * u + a_mu_5
        var = a_var_1 * tf.nn.sigmoid(a_var_2 * u + a_var_3) + a_var_4 * u + a_var_5

        z_recon = (z_noize - mu) * var + mu

    def _scaling_and_bias(self, x, name,
                              scope_name="scaling_and_bias"):
        """Scale and bias the input in BatchNorm

        The way to scale and bias is a bit different from the standard BatchNorm.
        Basically, we compute gamma * (z - mu) / std + beta,
        but here gamma * ((z - mu) / std + beta).
        Note the normalization (z - mu) / std is already computed as `x`.
                
        Parameters
        -----------------
        x: tf.Tensor
        name: str
        scope_name: str
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

        with tf.variable_scope(scope_name):
            v = self._get_variable_by_name(beta_name)
            beta = tf.get_variable(name=beta_name, shape=[depth]) \
              if v is None else v
            v = self._get_variable_by_name(gamma_name)
            gamma = tf.get_variable(name=gamma_name, shape=[depth]) \
              if v is None else v

        return gamma * (x - beta)

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

        return mean, tf.sqrt(var)

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
        
    def _compute_loss(self, ):
        """Compute loss op
        """
        loss = tf.nn.softmax_cross_entropy_with_logits(self.pred, self._y)
        self.loss = tf.reduce_mean(loss)

    def _accuracy(self, ):
        """Compute accuracy op
        """
        pred = self.pred
        y = self._y

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.accuracy = accuracy

    def _construct_ssl_ladder_network(self, x, y=None):
        """Construct SSL Ladder Network.

        If `y` is None, the reonstruction cost is only constructed;
        otherwise the classification loss is also constructed.

        Parameters
        -----------------
        x: tf.placeholder
            x is either labeled samples or unlabeled samples.
        y: tf.placeholder
            y is not None when x is labeled samples.

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
        lambda_list = [1] * self._L

        # Encoder
        h = x
        h_noise = h + tf.truncated_normal(h.get_shape())
        for i in range(self._L):
            # Clean encoder
            scope_linear = tf.variable_scope("linear")
            z_pre = self._linear(h, name="{}-th".format(i), self._n_dim, scope_linear)

            mu, std = self._moments(z_pre)
            z = self._batch_norm(z_pre, mu, std)

            scope_scaling_and_bias = tf.variable_scope("scaling_and_bias")
            h = tf.nn.tanh(self._scaling_and_bias(z, name="{}-th".format(i)),
                               scope_scaling_and_bias)

            # Append  
            mu_list.append(mu)
            std_list.append(std)
            z_list.append(z)
            
            # Corrupted encoder
            z_pre_noise = self._linear(h_noise,
                                       name="{}-th".format(i), self._n_dim, scope_linear)
            mu, std = self._moments(z_pre_noise)
            z_noize = self._batch_norm(z_pre_noise, mu, std) \
                      + tf.truncated_normal(z_pre_noise.get_shape())
            h_noise = tf.nn.tanh(self._scaling_and_bias(z_noise, name="{}-th".format(i)),
                                     scope_scaling_and_bias)

            # Append
            z_noise_list.append(z_noize)
            
        # Set classifier
        self.pred = h
            
        # Decoder
        for i in reversed(range(self._L)):
            if i == self._L - 1:
                mu, std = self._moments(h_noise)
                u = self._batch_norm(h_noise, mu, std)
            else:
                Vz = self._linear(z_recon, name="{}-th".format(i), self._n_dim)
                mu, std = self._moments(Vz)
                u = self._batch_norm(Vz, mu, std)

            z_noise = z_noise_list[i]
            z_recon = self._denoise(z_noize, u, name="{}-th".format(i))

            mu = mu_list[i]
            std = std_list[i]
            z_recon_bn = self._batch_norm(z_recon, mu, std)
            z_recon_bn_list.append(z_recon_bn)
            
        # Loss for both labeled and unlabeled samples
        C = 0
        for i in range(self._L):
            C += self.lambda_list[i] * (z_list[i] - z_recon_bn_list[i]) ** 2
            
        # Loss for labeled samples
        if y:
            C += tf.nn.softmax_cross_entropy(self.pred, self._y)

        return C

        
