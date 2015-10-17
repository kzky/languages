#!/usr/bin/env python

from model import BinaryClassifier
from model import Classifier
from scipy import sparse
import numpy as np
import scipy as sp
import logging
import model

class HPFSSLBinaryClassifier(BinaryClassifier):
    """
    Hyper Parameter Free Semi-Supervised Learning Binary Classifier.

    Original Binary Classifier, which update rule is as follows,
    
    m = \beta S X_l t
    S = (XLX^T + \beta X_l X_l^T) ^ {-1}
    L = X^{+} (S + mm^T)^{-1} X^T^{+})
    XLX^T = (S + mm^T)^{-1}
    \beta^{-1} = ( ||t - X_l^T m||_2^2 + \beta_{old}^{-1} Tr( I - SXLX^T) ) / l

    Notation is as follows,

    m: mean vector
    S: covariance matrix
    L: graph matrix
    beta: variance
    l: the number of labeled samples
    u: the number of unlabeled samples
    n: the number of samples
    t: labels
    X_l: labeled samples
    X_u: unlabeled samples
    I: identity matrix
    X^{+}: pseudo-inverse of X
    X^T: transopse of X
    Tr(X): trace of XS
    ||x||_2: L-2 norm
    x: one sample
    d: demension of x
    """

    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)
    logger = logging.getLogger("HPFSSLBinaryClassifier")
    
    def __init__(self, max_itr=100, threshold=1e-4,
                 learn_type=model.LEARN_TYPE_ONLINE,
                 ):
        """
        Arguments:
        - `max_itr`: max iteration for stopping criterion
        - `threshold`: threshold for stopping criterion
        """

        super(HPFSSLBinaryClassifier, self).__init__()
        
        self.max_itr = max_itr
        self.threshold = threshold
        self.learn_type = learn_type

        self.m = None
        self.S = None
        self.XLX = None
        self.beta = None

        self.is_sparse = None

    def learn(self, X_l, y, X_u):
        """
        
        Arguments:
        - `X_l`:
        - `y`:
        - `X_u`:
        """
        
        # dataset info
        self._set_data_info(X_l, y, X_u)

        # learn
        if self.learn_type == model.LEARN_TYPE_BATCH and self.is_sparse is False:
            self._learn_batch(self.X_l, self.y, self.X_u)
        elif self.learn_type == model.LEARN_TYPE_ONLINE and self.is_sparse is False:
            self._learn_online(self.X_l, self.y, self.X_u)
        elif self.is_sparse is True and self.learn_type == model.LEARN_TYPE_ONLINE:
            self._learn_online_in_sparse(self.X_l, self.y, self.X_u)
        else:
            raise Exception("learn_type %s in sparse=%s does not exist."
                            % (self.learn_type, self.is_sparse))

    def _set_data_info(self, X_l, y, X_u):
        """
        Initialize model.
        No need to add bias term into samples because here bias is added.

        Arguments:
        - `X_l`: samples, 2-d numpy array or sparse matrix
        - `y`: labels, 1-d numpy array, y is in {-1, 1}
        - `X_u`: unlabeled samples, 2-d numpy array
        """

        # set labels
        self._check_and_set_y(y)

        # dense case
        if not sp.sparse.issparse(X_l):
            self.is_sparse = False
            super(HPFSSLBinaryClassifier, self)._set_data_info(X_l, y, X_u)
            return

        # sparse case
        self.is_sparse = True
        if not sp.sparse.isspmatrix_csr(X_l):
            X_l = X_l.tocsr()
            X_u = X_u.tocsr()
            pass
        self.y = sp.sparse.csr_matrix(y)
        
        # set the number of labeled samples, unlabeled samples, and dimension.
        shape_l = X_l.shape
        shape_u = X_u.shape
        ds = [shape_l[1], shape_u[1]]
        idx = np.argmax(ds)
        d = ds[idx]
        self.d = d
        self.l = shape_l[0]

        self.u = shape_u[0]
        self.n = self.l + self.u
        if idx == 0:
            X_u = self._reshape(X_u, (X_u.shape[0], d))
        else:
            X_l = self._reshape(X_l, (X_l.shape[0], d))
        
        # set samples
        self.X_l = X_l
        self.X_u = X_u
        self.X = self.X_u
        

        # diagoanl matrix
        d = self.d
        self.I = sp.sparse.csr_matrix(sp.sparse.spdiags(np.ones(d), 0, d, d))

        pass

    def _reshape(self, X, shape):
        """Reshape the sparse matrix `a`.
     
        Returns a csr_matrix with shape `shape`.
        """
        X = X.tolil().reshape(shape).tocsr()
        return X
            
    def _learn_batch(self, X_l, y, X_u):
        """
        Learn in the batch fashion
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """
        
        # initialize L, beta, S, and m
        self.X_lX_l = self._compute_rank_one_sum()
        self.XLX = self.I
        self.beta = 1
        self.S = self._compute_S_batch()
        self.m = self._compute_m_batch()

        t = 0
        while (t < self.max_itr):
            t += 1
            # update parameters
            self.XLX = self._compute_XLX_batch()
            self.beta = self._compute_beta_batch()
            self.S = self._compute_S_batch()
            #m_old = self.m
            beta_old = self.beta
            self.m = self._compute_m_batch()

            # check stopping criteria
            if t % 100 == 0:
                self.logger.debug("itr: %d" % t)
                pass
            #if self._check_stopping_criteria_with_m(m_old):
            if self._check_stopping_criteria_with_beta(beta_old):
                break

        self.w = self.m
        
    def _learn_online(self, X_l, y, X_u):
        """
        Learn in the online fashion for S
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """

        # initialize L, beta, S, and m
        self.X_lX_l = self._compute_rank_one_sum()
        self.XLX = self.I
        self.beta = 1
        self.S = self._compute_initial_S_online()
        self.m = self._compute_m_online()

        t = 0
        while (t < self.max_itr):  # outer loop
            t += 1
            
            # update parameters
            self.XLX = self._compute_XLX_online()
            self.beta = self._compute_beta_online()
            #m_old = self.m
            beta_old = self.beta
            self.S = self._compute_S_online()
            self.m = self._compute_m_online()

            # check stopping criteria
            if t % 100 == 0:
                self.logger.debug("itr: %d" % t)
                pass
            #if self._check_stopping_criteria_with_m(m_old):
            if self._check_stopping_criteria_with_beta(beta_old):
                break
                
        self.w = self.m
        
    def _check_stopping_criteria_with_m(self, m_old):
        """
        
        Arguments:
        - `m_old`:
        """
        d = self.m - m_old
        d_L2_norm = np.sqrt(d.dot(d))
        self.logger.debug(d_L2_norm)
        if d_L2_norm < self.threshold:
            self.logger.debug("Norm of difference between the current m and previous m is %f" % d_L2_norm)
            return True
        else:
            return False

        pass

    def _check_stopping_criteria_with_beta(self, beta_old):
        """
        
        Arguments:
        - `m_old`:
        """
        d = self.beta - beta_old
        d_L2_norm = d ** 2
        self.logger.debug(d_L2_norm)
        if d_L2_norm < self.threshold:
            self.logger.debug("Norm of difference between the current m and previous m is %f" % d_L2_norm)
            return True
        else:
            return False
        
    def _compute_rank_one_sum(self, ):
        """
        Compute rank one sum.
        """
        X_l = self.X_l
        X_lX_l = X_l.T.dot(X_l)

        return X_lX_l

    def _compute_m_batch(self,):
        """
        Compute mean vector.
        """

        X_l = self.X_l
        y = self.y
        beta = self.beta
        S = self.S

        m = beta * S.dot(X_l.T.dot(y))
        return m

    def _compute_S_batch(self, ):
        """
        Compute covariance matrix
        
        """

        beta = self.beta
        XLX = self.XLX
        X_lX_l = self.X_lX_l

        S = np.linalg.inv(XLX + beta * X_lX_l)
        return S

    def _compute_XLX_batch(self,):
        """
        Compute L

        """
        m = self.m
        S = self.S

        # (n-by-d)-matrix
        XLX = np.linalg.inv(S + np.outer(m, m))

        return XLX

    def _compute_beta_batch(self, ):
        """
        Compute beta
        
        Arguments:
        - `m`:
        - `S`:
        - `beta_old`:
        """

        X_l = self.X_l
        y = self.y
        I = self.I
        m = self.m
        S = self.S
        XLX = self.XLX
        beta_old = self.beta
        l = self.l
        X_lX_l = self.X_lX_l
        SX_lX_l = S.dot(X_lX_l)

        residual = y - X_l.dot(m)
        inner_trace = I - S.dot(XLX)

        # Note that this is similar to RVM update rule and PSD holds.
        beta = l / (np.sum(residual ** 2) + inner_trace.trace() / beta_old)
        beta_2 = beta ** 2
        upper_bound = l / (SX_lX_l.diagonal() ** 2).sum()

        #beta = l / ((SX_lX_l.diagonal() ** 2).sum() + np.sum(residual ** 2))
        beta = beta if beta_2 < upper_bound else upper_bound
        return beta

    def _compute_m_online(self, ):
        """
        Compute m in an online fashion
        
        Arguments:
        - `x`: sample, 1-d numpy array, where x is a labeled sample
        - `y`: label
        
        """
        
        return self._compute_m_batch()

    def _compute_initial_S_online(self, ):
        """
        Compute S in an online fashion
        """
        
        XLX_inv = self.I

        # TODO: check whether to multiply small value should be better?
        S_t = XLX_inv

        for x in self.X_l:
            S_t = S_t - S_t.dot(np.outer(x, x)).dot(S_t) / (1 + x.dot(S_t).dot(x))
            pass

        return S_t

    def _compute_S_online(self, ):
        """
        Compute S in an online fashion
        """
        S = self.S
        m = self.m
        XLX_inv = S + np.outer(m, m)

        S_t = XLX_inv
        beta = self.beta

        for x in self.X_l:
            S_t = S_t - beta * S_t.dot(np.outer(x, x)).dot(S_t) / (1 + beta * x.dot(S_t).dot(x))
            pass

        return S_t

    def _compute_XLX_online(self, ):
        """
        """

        XLX = self.XLX
        beta = self.beta
        X_lX_l = self.X_lX_l
        S_inv = XLX + beta * X_lX_l
        m = self.m

        XLX = S_inv - S_inv.dot(np.outer(m, m)).dot(S_inv) / (1 + m.dot(S_inv).dot(m))

        return XLX
        
    def _compute_beta_online(self, ):
        """
        """
        return self._compute_beta_batch()


    """
    Functions for sparse dataset
    """
    def _learn_online_in_sparse(self, X_l, y, X_u):
        """
        Learn in the online fashion for S
        
        Arguments:
        - `X_l`: samples, sparse matrix in csr format
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """
        self.logger.debug("_learn_online_in_sparse start")
        
        # initialize L, beta, S, and m
        self.X_lX_l = self._compute_rank_one_sum_in_sparse()
        self.XLX = self.I
        self.beta = 1
        self.S = self._compute_initial_S_online_in_sparse()
        self.m = self._compute_m_online_in_sparse()

        t = 0
        while (t < self.max_itr):  # outer loop
            t += 1
            
            # update parameters
            self.XLX = self._compute_XLX_online_in_sparse()
            self.beta = self._compute_beta_online_in_sparse()
            m_old = self.m
            self.S = self._compute_S_online_in_sparse()
            self.m = self._compute_m_online_in_sparse()

            # check stopping criteria
            if t % 100 == 0:
                self.logger.debug("itr: %d" % t)
                pass
            if self._check_stopping_criteria_with_m_in_sparse(m_old):
                break
                
        self.w = self.m

        self.logger.debug("_learn_online_in_sparse finish")

    def _compute_rank_one_sum_in_sparse(self, ):
        """
        """
        X_lX_l = self._compute_rank_one_sum()
        return X_lX_l
        
    def _compute_XLX_online_in_sparse(self, ):
        """
        """

        XLX = self.XLX
        beta = self.beta
        X_lX_l = self.X_lX_l
        S_inv = XLX + beta * X_lX_l
        m = self.m

        denominator = 1 + m.dot(S_inv).dot(m.T).data[0]
        XLX = S_inv - S_inv.dot(m.T.dot(m)).dot(S_inv) / denominator
        
        return XLX

    def _compute_beta_online_in_sparse(self, ):
        """
        """
        X_l = self.X_l
        y = self.y
        I = self.I
        m = self.m
        S = self.S
        XLX = self.XLX
        beta_old = self.beta
        l = self.l
        
        residual = y - X_l.dot(m.T).T
        inner_trace = I - S.dot(XLX)
        trace = inner_trace.diagonal().sum()

        # Note that this is similar to RVM update rule and PSD holds.
        beta = l / (np.sum(residual.data[0] ** 2) + trace / beta_old)

        return beta

    def _compute_initial_S_online_in_sparse(self, ):

        XLX_inv = self.I
        S_t = XLX_inv

        for x in self.X_l:
            denominator = 1 + x.dot(S_t).dot(x.T).data[0]
            S_t = S_t - S_t.dot(x.T.dot(x)).dot(S_t) / denominator
            pass

        return S_t
        
    def _compute_S_online_in_sparse(self, ):
        """
        """
        S = self.S
        m = self.m
        XLX_inv = S + m.T.dot(m)

        S_t = XLX_inv
        beta = self.beta

        for x in self.X_l:
            denominator = 1 + beta * x.dot(S_t).dot(x.T).data[0]
            S_t = S_t - beta * S_t.dot(x.T.dot(x)).dot(S_t) / denominator
            pass

        return S_t

    def _compute_m_online_in_sparse(self, ):
        """
        """
        X_l = self.X_l
        y = self.y
        beta = self.beta
        S = self.S

        m = beta * y.dot(X_l).dot(S)
        return m

    def _check_stopping_criteria_with_m_in_sparse(self, m_old):
        """
        """
        d = self.m - m_old
        d_L2_norm = np.sqrt(d.dot(d.T).data[0])

        if d_L2_norm < self.threshold:
            self.logger.debug("Norm of difference between the current m and previous m is %f" % d_L2_norm)
            return True
        else:
            return False

        pass
    
        
class HPFSSLClassifier(Classifier):
    """
    HPFSSLClassifier handles multi-class with HPFSSLBinaryClassifier.

    This class JUST coordinates binary classifiers for handling muti-classes.
    """
    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)
    logger = logging.getLogger("HPFSSLClassifier")

    def __init__(self,
                 multi_class=model.MULTI_CLASS_ONE_VS_ONE,
                 max_itr=10000, threshold=1e-4,
                 learn_type=model.LEARN_TYPE_ONLINE,
                 ):
        """
        """
        
        super(HPFSSLClassifier, self).__init__(
            multi_class=multi_class,
        )

        self.max_itr = max_itr
        self.threshold = threshold
        self.learn_type = learn_type
        
        self.logger.debug(
            "Parameters set with max_itr = %d, \
            threshold = %f, \
            multi_class = %s, \
            learn_type = %s" %
            (self.max_itr, self.threshold, self.multi_class, self.learn_type))

    def create_binary_classifier(self, ):
        """
        """
        binary_classifier = HPFSSLBinaryClassifier(
            max_itr=self.max_itr, threshold=self.threshold,
            learn_type=self.learn_type
        )
        return binary_classifier
        
    def _create_classifiers(self, param_grid=[{}]):
        """
        
        Arguments:
        - `param_grid`:
        """
        classifiers = []
        for param in param_grid:
            max_itr = param["max_itr"]
            threshold = param["threshold"]
            learn_type = param["learn_type"]

            classifier = HPFSSLClassifier(
                max_itr=max_itr,
                threshold=threshold,
                learn_type=learn_type
            )
            classifiers.append(classifier)

        return classifiers

def main():
    import time
    from sklearn.metrics import confusion_matrix
    
    # labeled sample
    l_data_path = "/home/kzk/datasets/uci_csv_ssl_lrate_fixed_1_50_1_98/car/5_l.csv"
    data_l = np.loadtxt(l_data_path, delimiter=" ")
    data_l = np.hstack((data_l, np.reshape(np.ones(data_l.shape[0]), (data_l.shape[0], 1))))
    y_l = data_l[:, 0]
    X_l = data_l[:, 1:]

    # unlabeled sample
    u_data_path = "/home/kzk/datasets/uci_csv_ssl_lrate_fixed_1_50_1_98/car/5_u.csv"
    data_u = np.loadtxt(u_data_path, delimiter=" ")
    data_u = np.hstack((data_u, np.reshape(np.ones(data_u.shape[0]), (data_u.shape[0], 1))))
    X_u = data_u[:, 1:]

    # test sample
    t_data_path = "/home/kzk/datasets/uci_csv_ssl_lrate_fixed_1_50_1_98/car/5_t.csv"
    data_t = np.loadtxt(t_data_path, delimiter=" ")
    data_t = np.hstack((data_t, np.reshape(np.ones(data_t.shape[0]), (data_t.shape[0], 1))))
    y_t = data_t[:, 0]
    X_t = data_t[:, 1:]
    
    # learn
    st = time.time()
    model = HPFSSLClassifier(
        max_itr=100, threshold=1e-4,
        learn_type="batch", multi_class="ovo")
    model.learn(X_l, y_l, X_u)
    et = time.time()
    print "Elapsed time: %f [s]" % (et - st)

    # predict
    outputs = []
    for i, x in enumerate(X_t):
        outputs_ = model.predict(x)
        outputs.append(outputs_[0][0])

    # confusion matrix
    cm = confusion_matrix(y_t, outputs)
    print cm
    print 100.0 * np.sum(cm.diagonal())/len(y_t)


def sparse_main():
    from sklearn.datasets import load_svmlight_file
    from sklearn.metrics import confusion_matrix
    import time

    # data
    data_path = "/home/kzk/datasets/news20/news20.dat"
    (X, y) = load_svmlight_file(data_path)
    n = X.shape[0]
    X = sp.sparse.hstack((X, sp.sparse.csr_matrix(np.reshape(np.ones(n), (n, 1)))))
    X_l = sp.sparse.csr_matrix(X)
    X_u = sp.sparse.csr_matrix(X)

    st = time.time()
    # learn
    model = HPFSSLClassifier(max_itr=0, threshold=1e-4, learn_type="online", multi_class="ovo")
    model.learn(X_l, y, X_u)
    et = time.time()
    print "Elapsed time: %f [s]" % (et - st)
    
    # predict
    outputs = []
    for i, x in enumerate(X):
        outputs_ = model.predict(x)
        outputs.append(outputs_[0][0])

    # confusion matrix
    cm = confusion_matrix(y, outputs)
    print cm
    print 100.0 * np.sum(cm.diagonal())/len(y)
    
if __name__ == '__main__':
    main()
    #sparse_main()
