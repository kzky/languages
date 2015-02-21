#!/usr/bin/env python

from model import Model
import numyp as np


class TPFNSOriginalBinaryClassifier(Model):
    """
    Tuning Parameter Free No-Scaling Semi-Supervised Learning.

    Original Binary Classifier, which update rule is as follows,
    
    m = beta S X_l t
    S = (I + X L X^T + beta X_l X_l^T) ^ {-1}
    L = X^{+} ((XX^T)^{-1} + mm^T + S) X^T^{+})
    beta^{-1} = ( ||t - X_l^T m||_2^2 + beta_old^{-1} Tr( I - S (I + XLX^T)) ) / l

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

    def __init__(self, max_itr=1000, threshold=1e-6):
        """
        Arguments:
        - `max_itr`: max iteration for stopping criterion
        - `threshold`: threshold for stopping criterion

        """

        super(TPFNSOriginalBinaryClassifier, self).function()

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

    def learn(self, X_l, y, X_u):
        """
        Learn
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """

        self._init_model(X_l, y, X_u)

    def _init_model(self, X_l, y, X_u):
        """
        Initialize model.

        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """

        # set  dataset
        self.X_l = X_l
        self.y = y
        self.X_u = X_u
        self.X = np.vstack((X_l, X_u))

        # set the number of labeled samples and unlabeled samples
        shape_l = self.X_l.shape
        self.d = shape_l[1]
        self.l = shape_l[0]

        shape_u = self.X_u.shape
        self.u = shape_u[0]
        self.n = self.l + self.u

        # diagoanl matrix
        self.I = np.diag(np.ones(self.d))

        # initialize L, beta, S, and m
        self.L = self.I
        self.beta = 1
        self._compute_S(self.L, self.beta)
        self._compute_m(self.S, self.beta)

        pass

    def _learn_internally(self, m, S, L, beta):
        """
        
        Arguments:
        - `m`:
        - `S`:
        - `L`:
        - `beta`:
        """

        t = 0
        while (t < self.max_itr):  # check stopping criteria

            # update parameters
            self.L = self._compute_L()
            self.beta = self._compute_beta()
            self.S = self._compute_S()

            beta_old = self.beta
            self.beta = self._compute_beta(beta_old)

            # check stopping criteria
            if self._check_stopping_criteria(beta_old):
                break
            
            t += 1
        pass

    def _check_stopping_criteria(self, beta_old):
        """
        Check stopping criteria
        
        Arguments:
        - `beta_old`:
        """

        d = self.beta - beta_old
        d_L2_2 = np.sqrt(d ** 2)

        if d_L2_2 < self.threshold:
            return True
        else:
            return False

    def _compute_m(self,):
        """
        Compuate mean vector.
        """

        X_l = self.X_l
        y = self.y
        beta = self.beta
        S = self.S
        
        m = beta * S.dot(y).dot(X_l)
        return m

    def _compute_S(self, ):
        """
        Update covariance matrix
        
        """

        X_l = self.X_l
        X = self.X
        beta = self.beta
        L = self.L
        I = self.I

        S = I + X.T.dot(L.dot(X)) + beta * X_l.T.dot(X_l)
        return S
                                    
    def _compute_L(self,):
        """
        Update L

        """
        X = self.X
        m = self.m
        S = self.S
        inner_tem = np.linalg.inv(X.T.dot(X)) + m.dot(m) + S

        # (n-by-d)-matrix
        p_inv_X = self._compute_full_row_rank_pseudo_inverse(X)

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

    def _compute_beta(self, ):
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
        inner_trace = I - S.dot(I + X.T.dot(L).dot(X))

        # Note that this is similar to RVM update rule and PSD holds.
        beta = l / (residual ** 2 + inner_trace.trace() / beta_old)

        return beta
        
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
