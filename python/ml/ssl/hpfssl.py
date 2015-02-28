#!/usr/bin/env python

from model import Model
import numyp as np


class HPFSSLBinaryClassifier(Model):
    """
    Hyper Parameter Free Semi-Supervised Learning Binary Classifier.

    Original Binary Classifier, which update rule is as follows,
    
    m = \beta S X_l t
    S = (XLX^T + \beta X_l X_l^T) ^ {-1}
    L = X^{+} (S + mm^T) X^T^{+})
    \beta^{-1} = ( ||t - X_l^T m||_2^2 + \beta_{old}^{-1} Tr( I - SXLX^T) ) / l

    Notation is as follows,
    
    m: mean vector
    S: covariance matrix
    L: graph matrix
    beta: variance
    l: the number of labeld samples
    u: the number of unlabeld samples
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

    LEARN_TYPE_BATCH = "batch"
    LEARN_TYPE_SEMI_ONLINE = "semi_online"
    LEARN_TYPE_ONLINE = "online"

    def __init__(self, max_itr=100, threshold=1e-6, learn_type="batch"):
        """
        Arguments:
        - `max_itr`: max iteration for stopping criterion
        - `threshold`: threshold for stopping criterion
        """

        super(HPFSSLBinaryClassifier, self).function()

        self.max_itr = max_itr
        self.threshold = threshold

        self.X_l = None
        self.y = None
        self.X_u = None
        self.X = None
        self.m = None
        self.S = None
        self.L = None
        self.beta = None

        if learn_type == HPFSSLBinaryClassifier.LEARN_TYPE_BATCH:
            self.learn = self._learn_batch
        elif learn_type == HPFSSLBinaryClassifier.LEARN_TYPE_SEMI_ONLINE:
            self.learn = self._learn_semi_online
        elif learn_type == HPFSSLBinaryClassifier.LEARN_TYPE_ONLINE:
            self.learn = self._learn_online
        else:
            self.learn = self._learn_batch

    def _init_model(self, X_l, y, X_u):
        """
        Initialize model.
        No need to add bias term into samples because here bias is added.

        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array, y is in {-1, 1}
        - `X_u`: unlabeled samples, 2-d numpy array
        """
        # set the number of labeled samples and unlabeled samples
        shape_l = X_l.shape
        self.d = shape_l[1] + 1
        self.l = shape_l[0]

        shape_u = X_u.shape
        self.u = shape_u[0]
        self.n = self.l + self.u

        # set dataset and add bias
        self.X_l = np.hstack((self.X_l, np.reshape(np.ones(self.l), (self.l, 1))))
        self.y = y
        self.X_u = np.hstack((self.X_u, np.reshape(np.ones(self.u), (self.u, 1))))
        self.X = np.vstack((X_l, X_u))

        # diagoanl matrix
        self.I = np.diag(np.ones(self.d))

        # initialize L, beta, S, and m
        self.L = self.I
        self.beta = 1
        self._compute_S()
        self._compute_m()

        pass

    def _learn_batch(self, X_l, y, X_u):
        """
        Learn in the batch way
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """

        self._init_model(X_l, y, X_u)

        t = 0
        while (t < self.max_itr):  # check stopping criteria

            # update parameters
            self.L = self._compute_L_batch()
            self.beta = self._compute_beta_batch()
            self.S = self._compute_S_batch()
            m_old = self.m
            self.m = self._compute_m_batch()

            # check stopping criteria
            if self._check_stopping_criteria_with_m(m_old):
                break
                
            t += 1
        pass

    def _learn_semi_online(self, X_l, y, X_u):
        """
        Learn in the semi-online way
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array

        """

        self._init_model(X_l, y, X_u)

        t = 0
        while (t < self.max_itr):  # outer loop
            t += 1
            
            # update parameters
            self.L = self._compute_L_batch()
            self.beta = self._compute_beta_batch()

            m_old = self.m
            for x in self.X_l:
                self.S = self._compute_S_online()
                self.m = self._compute_m_online()

            # check stopping criteria
            if self._check_stopping_criteria_with_m(m_old):
                break
        
        pass

    def _learn_online(self, X_l, y, X_u):
        """
        Learn in the online way
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """
        self._init_model(X_l, y, X_u)

        t = 0
        while (t < self.max_itr):  # outer loop
            t += 1
            m_old = self.m
            for x in self.X_l:
                # update parameters
                self.L = self._compute_L_batch()
                self.beta = self._compute_beta_batch()
                self.S = self._compute_S_online()
                self.m = self._compute_m_online()

            # check stopping criteria
            if self._check_stopping_criteria_with_m(m_old):
                break
        
        pass
        
    def _check_stopping_criteria_with_m(self, m_old):
        """
        
        Arguments:
        - `m_old`:
        """
        d = self.m - m_old
        d_L2_norm = np.sqrt(d**2)
        
        if d_L2_norm < self.threshold:
            return True
        else:
            return False

        pass

    def _compute_m_batch(self,):
        """
        Compute mean vector.
        """

        X_l = self.X_l
        y = self.y
        beta = self.beta
        S = self.S
        
        m = beta * S.dot(y).dot(X_l)
        return m

    def _compute_S_batch(self, ):
        """
        Update covariance matrix
        
        """

        X_l = self.X_l
        X = self.X
        beta = self.beta
        L = self.L

        S = X.T.dot(L.dot(X)) + beta * X_l.T.dot(X_l)
        return S
                                    
    def _compute_L_batch(self,):
        """
        Update L

        """
        X = self.X
        m = self.m
        S = self.S

        # (n-by-d)-matrix
        p_inv_X = self._compute_full_row_rank_pseudo_inverse(X)
        inner_tem = S + m.dot(m)
        L = p_inv_X.dot(np.linalg.inv(inner_tem)).dot(p_inv_X.T)
        
        return L

    def _compute_full_row_rank_pseudo_inverse(self, X):
        """
        Compute pseudo inverse of matrix X.
        Note in the mathmatical notation, X is (d-by-n)-matrix;
        however X is now (n-by-d)-matirx, so take transpose of X first.

        Dimension of return matrix is (n-by-d)-matrix.
        Arguments:
        - X: X in R^{n by d}
        
        """

        Z = X.T
        
        return Z.T.dot(np.linalg.inv(Z.dot(Z.T)))

    def _compute_beta_batch(self, ):
        """
        Update beta
        
        Arguments:
        - `m`:
        - `S`:
        - `beta_old`:
        """

        X_l = self.X_l
        y = self.y
        X = self.X
        I = self.I
        m = self.m
        S = self.S
        L = self.L
        beta_old = self.beta
        l = self.l
        
        residual = y - X_l.dot(m)
        inner_trace = I - S.dot(X.T.dot(L).dot(X))

        # Note that this is similar to RVM update rule and PSD holds.
        beta = l / (residual ** 2 + inner_trace.trace() / beta_old)

        return beta

    def _compute_m_online(self, x, y):
        """
        Compute m in an online way
        
        Arguments:
        - `x`: sample, 1-d numpy array, where x is a labeled sample
        - `y`: label
        
        """
        m = self.beta * self.S.dot(x) * y
        return m

    def _compute_S_online(self, x, y):
        """
        Compute m in an online way
        
        Arguments:
        - `x`: sample, 1-d numpy array, where x is a labeled sample
        - `y`: label

        """
        S_t = self.S

        # rank-one update
        S = S_t - (S_t.dot(x.T.dot(x)).dot(S_t)) / (1 + self.beta * x.dot(S_t).dot(x))
        return S
        
    def _compute_L_online(self, ):
        """
        """
        self._compute_L_batch()

    def _compute_beta_online(self, ):
        """
        """
        self._compute_beta_batch

        
    def predict(self, x):
        """
        Predict label for x
        
        Arguments:
        - `x`: sample, 1-d numpy array
        """
        
        pass

    def _normalize(self, X):
        """
        Nomalize dataset using some methods,
        e.g., zero mean, unit variance.

        Arguments:
        - `X`: all samples, 2-d numpy array

        """
        pass
        
def main():

    pass

if __name__ == '__main__':
    main()
