#!/usr/bin/env python

from model import BinaryClassifier
from model import Classifier
from sklearn.metrics import confusion_matrix
import numpy as np
import logging
import model


class RVMBinaryClassifier(BinaryClassifier):
    """
    Relevance Vector Machine Binary Classifier.

    Original Binary Classifier, which update rule is as follows,
    
    m = \beta S X t
    S = (\beta X X^T) ^ {-1}
    L = (S + mm^T)^{-1}
    \beta^{-1} = ( ||t - X^T m||_2^2 + \beta_{old}^{-1} Tr( I - SA^T) ) / l

    Notation is as follows,

    m: mean vector
    S: covariance matrix
    alphas: covariances for normal distributions
    beta: variance
    l: the number of labeled samples
    n: the number of samples
    t: labels
    X: labeled samples
    I: identity matrix
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
    logger = logging.getLogger("RVMBinaryClassifier")
    
    def __init__(self, max_itr=100, threshold=1e-4,
                 learn_type=model.LEARN_TYPE_ONLINE,
                 alpha_threshold=1e-24,
                 ):
        """
        Arguments:
        - `max_itr`: max iteration for stopping criterion
        - `threshold`: threshold for stopping criterion
        """

        super(RVMBinaryClassifier, self).__init__()
        
        self.max_itr = max_itr
        self.threshold = threshold
        self.learn_type = learn_type
        self.alpha_threshold = alpha_threshold

        self.m = None
        self.S = None
        self.alphas = None
        self.beta = None

    def learn(self, X, y):
        """
        
        Arguments:
        - `X`:
        - `y`:
        """
        
        # dataset info
        self._set_data_info(X, y)

        self.XX = self._compute_rank_one_sum()

        # learn
        if self.learn_type == model.LEARN_TYPE_BATCH:
            self._learn_batch(self.X, self.y)
        elif self.learn_type == model.LEARN_TYPE_ONLINE:
            self._learn_online(self.X, self.y)
        else:
            raise Exception("learn_type %s does not exist." % self.learn_type)

    def _learn_batch(self, X, y):
        """
        Learn in the batch fashion
        
        Arguments:
        - `X`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        """
        
        # initialize L, beta, S, and m
        self.alphas = np.ones(self.d)
        self.beta = 1
        self.S = self._compute_S_batch()
        self.m = self._compute_m_batch()

        t = 0
        while (t < self.max_itr):
            t += 1
            # update parameters
            self.alphas = self._compute_alphas_batch()
            self.beta = self._compute_beta_batch()
            self.S = self._compute_S_batch()
            m_old = self.m
            self.m = self._compute_m_batch()

            # check stopping criteria
            if t % 100 == 0:
                self.logger.debug("itr: %d" % t)
                pass
            if self._check_stopping_criteria_with_m(m_old):
                break

        self.w = self.m

    def _learn_online(self, X, y):
        """
        Learn in the online fashion
        
        Arguments:
        - `X`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        """

        # initialize alphas, beta, S, and m
        self.alphas = np.ones(self.d)
        self.beta = 1
        self.S = self._compute_initial_S_online()
        self.m = self._compute_m_online()

        t = 0
        while (t < self.max_itr):  # outer loop
            t += 1
            
            # update parameters
            self.alphas = self._compute_alphas_online()
            self.beta = self._compute_beta_online()
            m_old = self.m
            self.S = self._compute_S_online()
            self.m = self._compute_m_online()

            # check stopping criteria
            if t % 100 == 0:
                self.logger.debug("itr: %d" % t)
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
            self.logger.debug("Norm of difference between the current m and previous m is %f" % d_L2_norm)
            return True
        else:
            return False

        pass

    def _compute_rank_one_sum(self, ):
        """
        Compute rank one sum.
        """
        X = self.X
        XX = X.T.dot(X)

        return XX

    def _compute_m_batch(self,):
        """
        Compute mean vector.
        """

        X = self.X
        y = self.y
        beta = self.beta
        S = self.S
        
        m = beta * S.dot(X.T.dot(y))
        return m

    def _compute_S_batch(self, ):
        """
        Compute covariance matrix
        
        """

        beta = self.beta
        XX = self.XX
        alphas = self.alphas
        A = np.diag(alphas)

        S = np.linalg.inv(A + beta * XX)
        return S

    def _compute_alphas_batch(self,):
        """
        Compute L

        """
        S = self.S
        alphas_old = self.alphas
        ses = np.diagonal(S)
        gammas = 1 - alphas_old * ses
        m = self.m

        alphas = None
        try:
            alphas = gammas / (m ** 2)
        except Exception:
            alphas = alphas_old
        
        # numerical check
        alpha_threshold = self.alpha_threshold
        indices = np.where(alphas < alpha_threshold)[0]
        alphas[indices] = alpha_threshold
        
        return alphas

    def _compute_beta_batch(self, ):
        """
        Compute beta
        
        Arguments:
        - `m`:
        - `S`:
        - `beta_old`:
        """

        X = self.X
        y = self.y
        
        m = self.m
        S = self.S
        ses = np.diagonal(S)
        alphas = self.alphas
        beta_old = self.beta
        l = self.l
        
        residual = y - X.dot(m)
        gammas = 1 - ses.dot(alphas)

        # Note that this is similar to RVM update rule and PSD holds.
        beta = l / (np.sum(residual ** 2) + np.sum(gammas) / beta_old)

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
        
        A_inv = self.I
        S_t = A_inv

        for x in self.X:
            S_t = S_t - S_t.dot(np.outer(x, x)).dot(S_t) / (1 + x.dot(S_t).dot(x))
            pass

        return S_t

    def _compute_S_online(self, ):
        """
        Compute S in an online fashion
        """

        alphas = self.alphas
        A_inv = np.diag(1 / alphas)

        S_t = A_inv
        beta = self.beta

        for x in self.X:
            S_t = S_t - beta * S_t.dot(np.outer(x, x)).dot(S_t) / (1 + beta * x.dot(S_t).dot(x))
            pass

        return S_t

    def _compute_alphas_online(self, ):
        """
        """

        alphas = self._compute_alphas_batch()
        return alphas
        
    def _compute_beta_online(self, ):
        """
        """
        return self._compute_beta_batch()

        
class RVMClassifier(Classifier):
    """
    RVMClassifier handles multi-class with RVMBinaryClassifier.

    This class JUST coordinates binary classifiers for handling muti-classes.
    """
    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)
    logger = logging.getLogger("RVMClassifier")

    def __init__(self,
                 multi_class=model.MULTI_CLASS_ONE_VS_ONE,
                 max_itr=100, threshold=1e-4,
                 learn_type=model.LEARN_TYPE_ONLINE,
                 alpha_threshold=1e-24,
                 ):
        """
        """
        
        super(RVMClassifier, self).__init__(
            multi_class=multi_class,
        )

        self.max_itr = max_itr
        self.threshold = threshold
        self.learn_type = learn_type
        self.alpha_threshold = alpha_threshold
        
        self.logger.debug(
            "Parameters set with max_itr = %d, threshold = %f, multi_class = %s, learn_type = %s" %
            (self.max_itr, self.threshold, self.multi_class, self.learn_type))

    def create_binary_classifier(self, ):
        """
        """
        binary_classifier = RVMBinaryClassifier(
            max_itr=self.max_itr, threshold=self.threshold,
            learn_type=self.learn_type
        )
        return binary_classifier


def main():

    # data
    data_path = "/home/kzk/datasets/uci_csv/iris.csv"
    data = np.loadtxt(data_path, delimiter=" ")
    y = data[:, 0]
    X = data[:, 1:]
    n = X.shape[0]
    X = np.hstack((X, np.reshape(np.ones(n), (n, 1))))

    # learn
    model = RVMClassifier(max_itr=50, threshold=1e-4, learn_type="batch", multi_class="ovo", alpha_threshold=1e-24,)
    model.learn(X, y)

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
