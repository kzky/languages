#!/usr/bin/env python

from model import BinaryClassifier
from model import Classifier
from sklearn.metrics import confusion_matrix
import numpy as np
import logging
import model

class LapRLSBinaryClassifier(BinaryClassifier):
    """
    Laplacian Regularized Least Square Bninary Classifier.
    
    Notation is as follows,

    X_l: labeled samples
    y_l: labels
    X_u: unlabeled samples
    w: weights of model
    labmda: balancing parameter between loss function and regularizer, attached to loss term
    sigma_s: parameter of graph laplacian
    """
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("LapRLSBinaryClassifier")

    def __init__(self, lam=1, sigma_s=1, **kargs):
        """
        Arguments:
        - `lam`: lambda, balancing parameter between loss function and regularizer, attached to loss term
        - `sigma_s`: parameter of graph laplacian

        """
        super(LapRLSBinaryClassifier, self).__init__(
            lam=lam, sigma_s=sigma_s
        )
        
        self.lam = lam
        self.sigma_s = sigma_s

        pass

    def learn(self, X_l, y, X_u):
        """
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array

        """

        self._init_model(X_l, y, X_u)
        self.X_lX_l = self._compute_rank_one_sum()
        
        self._learn_batch(X_l, y, X_u)

        pass
        
    
    def _learn_batch(self, X_l, y, X_u):
        """
        Learn in the batch fashion
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """

        # component of w
        X_lX_l = self.X_lX_l
        X_l = self.X_l
        X = self.X
        y = self.y
        kernel = self._rbf
        L = self._compute_L(kernel=kernel)
        XLX = X.T.dot(L).dot(X)
        I = self.I
        lam = self.lam

        inv_inner_term = lam * X_lX_l + I + XLX
        w = lam * np.linalg.inv(inv_inner_term).dot(X_l.T).dot(y)
        
        self.w = w

    def predict(self, x):
        """
        
        Arguments:
        - `x`: one samples with d-dimension
        """
        x = np.hstack((x, 1))
        w = self.w

        return w.dot(x)
    
    def _compute_rank_one_sum(self, ):
        """
        Compute rank one sum.
        """
        X_l = self.X_l
        X_lX_l = X_l.T.dot(X_l)

        return X_lX_l

    def _compute_L(self, kernel):
        """
        Compute graph laplacian

        Now support only rbf kernel
        """

        n = self.n
        L = np.zeros((n, n))
        X = self.X

        for i in xrange(0, n):
            x_i = X[i, :]
            for j in xrange(0, n):
                x_j = X[j, :]
                if i <= j:
                    L[i, j] = kernel(x_i, x_j)
                else:
                    L[i, j] = L[j, i]
            pass
        return L
        

    def _rbf(self, x, y):
        """
        
        Arguments:
        - `x`:
        - `y`:
        """
        
        sigma_s_2 = self.sigma_s ** 2
        diff = x - y
        norm_L2_2 = diff.dot(diff)
        val = np.exp(- norm_L2_2 / sigma_s_2)

        return val


class LapRLSClassifier(Classifier):
    """
    Laplacian Regularized Least Square Bninary Classifier.

    This class JUST coordinates binary classifiers for handling muti-classes.
    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("LapRLSClassifier")

    def __init__(self,
                 multi_class=model.MULTI_CLASS_ONE_VS_ONE,
                 lam=1,
                 sigma_s=1,
                 kernel="",
                 **kargs):
        """
        """

        super(LapRLSClassifier, self).__init__(
            multi_class=multi_class,
            lam=lam,
            sigma_s=sigma_s
        )

        self.logger.info("Parameters set with lambda = %f, sigma = %f, multi_class is %s" % (self.lam, self.sigma_s, self.multi_class))

        self.internal_classifier = LapRLSBinaryClassifier
        
def main():

    # data
    data_path = "/home/kzk/datasets/uci_csv/glass.csv"
    data = np.loadtxt(data_path, delimiter=" ")
    y = data[:, 0]
    X = data[:, 1:]

    # learn
    lam = 100
    sigma_s = 0.01
    model = LapRLSClassifier(lam=lam, sigma_s=sigma_s, multi_class="ovo")
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
