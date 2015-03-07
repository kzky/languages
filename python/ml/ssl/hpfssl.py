#!/usr/bin/env python

from model import BinaryClassifier
from model import Classifier
from sklearn.metrics import confusion_matrix
import numpy as np
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

    logging.basicConfig(level=logging.DEBUG)
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

    def learn(self, X_l, y, X_u):
        """
        
        Arguments:
        - `X_l`:
        - `y`:
        - `X_u`:
        """
        
        # dataset info
        self._set_data_info(X_l, y, X_u)

        # compute X_l.T * X_l
        self.X_lX_l = self._compute_rank_one_sum()

        # learn
        if self.learn_type == model.LEARN_TYPE_BATCH:
            self._learn_batch(self.X_l, self.y, self.X_u)
        elif self.learn_type == model.LEARN_TYPE_ONLINE:
            self._learn_online(self.X_l, self.y, self.X_u)
        else:
            raise Exception("learn_type %s does not exist." % self.learn_type)

    def _learn_batch(self, X_l, y, X_u):
        """
        Learn in the batch fashion
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """
        
        # initialize L, beta, S, and m
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
            m_old = self.m
            self.m = self._compute_m_batch()

            # check stopping criteria
            if t % 100 == 0:
                self.logger.info("itr: %d" % t)
                pass
            if self._check_stopping_criteria_with_m(m_old):
                break

        self.w = self.m

    def _learn_online(self, X_l, y, X_u):
        """
        Learn in the online fashion
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """

        # initialize L, beta, S, and m
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
            m_old = self.m
            self.S = self._compute_S_online()
            self.m = self._compute_m_online()

            # check stopping criteria
            if t % 100 == 0:
                self.logger.info("itr: %d" % t)
                pass
            if self._check_stopping_criteria_with_m(m_old):
                break
                
        self.w = self.m

    def _check_stopping_criteria_with_m(self, m_old):
        """
        
        Arguments:
        - `m_old`:
        """
        d = self.m - m_old
        d_L2_norm = np.sqrt(d.dot(d))
        if d_L2_norm < self.threshold:
            self.logger.info("Norm of difference between the current m and previous m is %f" % d_L2_norm)
            return True
        else:
            return False

        pass

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
        
        residual = y - X_l.dot(m)
        inner_trace = I - S.dot(XLX)

        # Note that this is similar to RVM update rule and PSD holds.
        beta = l / (np.sum(residual ** 2) + inner_trace.trace() / beta_old)

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

        
    def predict(self, x):
        """
        Predict label for x
        
        Arguments:
        - `x`: sample, 1-d numpy array
        """

        x = np.hstack((x, 1))
        
        return self.m.dot(x)
    
class HPFSSLClassifier(Classifier):
    """
    HPFSSLClassifier handles multi-class with HPFSSLBinaryClassifier.

    This class JUST coordinates binary classifiers for handling muti-classes.
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("HPFSSLClassifier")

    def __init__(self,
                 multi_class=model.MULTI_CLASS_ONE_VS_ONE,
                 max_itr=100, threshold=1e-4,
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
        
        self.logger.info("Parameters set with max_itr = %d, threshold = %f, multi_class = %s, learn_type = %s" %
                         (self.max_itr, self.threshold, self.multi_class, self.learn_type))

    def create_intrenal_classifier(self, ):
        """
        """
        internal_classifier = HPFSSLBinaryClassifier(
            max_itr=self.max_itr, threshold=self.threshold,
            learn_type=self.learn_type
        )
        return internal_classifier
        
def main():

    # data
    data_path = "/home/kzk/datasets/uci_csv/glass.csv"
    data = np.loadtxt(data_path, delimiter=" ")
    y = data[:, 0]
    X = data[:, 1:]

    # learn
    model = HPFSSLClassifier(max_itr=50, threshold=1e-4, learn_type="online", multi_class="ovo")
    model.learn(X, y, X)

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
