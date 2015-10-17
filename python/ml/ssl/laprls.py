#!/usr/bin/env python

from model import BinaryClassifier
from model import Classifier
from sklearn.metrics import confusion_matrix
import numpy as np
import logging
import model
import time


class LapRLSBinaryClassifier(BinaryClassifier):
    """
    Laplacian Regularized Least Square Bninary Classifier.
    
    Notation is as follows,

    X_l: labeled samples
    y_l: labels
    X_u: unlabeled samples
    w: weights of model
    labmda: balancing parameter between loss function and regularizer, attached to loss term
    gamma_s: parameter of graph laplacian
    """
    
    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)
    logger = logging.getLogger("LapRLSBinaryClassifier")

    def __init__(self, lam=1, normalized=True,
                 gamma_s=1, kernel=model.KERNEL_RBF):
        """
        Arguments:
        - `lam`: lambda, balancing parameter between loss function and regularizer, attached to loss term
        - `gamma_s`: parameter of graph laplacian
        - `normalized:` graph laplacian is normalized if true; otherwise unnomalized
        - `kernel:` kernel to be used
        """
        super(LapRLSBinaryClassifier, self).__init__()
        
        self.lam = lam
        self.gamma_s = gamma_s
        self.normalized = normalized

        self._set_kernel(kernel)

        pass

    def learn(self, X_l, y, X_u):
        """
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array

        """

        # dataset info
        self._set_data_info(X_l, y, X_u)

        # compute X_l.T * X_l
        self.X_lX_l = self._compute_rank_one_sum()

        # learn batch
        self._learn_batch(self.X_l, self.y, self.X_u)

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
        L = self._compute_L()
        XLX = X.T.dot(L).dot(X)
        I = self.I
        lam = self.lam

        inv_inner_term = lam * X_lX_l + I + XLX

        w = lam * np.linalg.inv(inv_inner_term).dot(X_l.T).dot(y)
        self.w = w

    def _compute_rank_one_sum(self, ):
        """
        Compute rank one sum.
        """
        X_l = self.X_l
        X_lX_l = X_l.T.dot(X_l)

        return X_lX_l

    def _compute_L(self, ):
        """
        Compute graph laplacian

        Now support only rbf kernel
        """

        X = self.X
        W = self._compute_L_with_rbf(X)

        self.logger.info("compute_L starts")
        s = time.time()

        L = None
        d = np.sum(W, axis=0)
        if self.normalized:
            I = np.ones(len(d))
            D_norm = np.diag(1 / np.sqrt(d))
            L = I - D_norm.dot(W).dot(D_norm)
        else:
            D = np.diag(d)
            L = D - W

        e = time.time()
        self.logger.info("compute_L finish with %f [s]" % (e - s))

        return L

    def _compute_L_with_rbf(self, X):
        """
        """
        H = np.tile(np.diag(np.dot(X, X.T)), (X.shape[0], 1))
        G = np.dot(X, X.T)
        distmat = H - 2 * G + H.T
        gamma_s = self.gamma_s
        
        K = np.exp(-gamma_s * distmat)
        return K

    def _set_kernel(self, kernel=model.KERNEL_RBF):
        """
        
        Arguments:
        - `kernel`:
        """

        if kernel == model.KERNEL_RBF:
            self.kernel = self._rbf
        elif kernel == model.KERNEL_LINEAR:
            self.kernel = self._linear
        else:
            raise Exception("%s kernel does not exist" % (kernel))
        
        pass
        
    def _rbf(self, x, y):
        """
        
        Arguments:
        - `x`:
        - `y`:
        """
        
        gamma_s = self.gamma_s
        diff = x - y
        norm_L2_2 = diff.dot(diff)
        val = np.exp(- gamma_s * norm_L2_2)

        return val

    def _linear(self, x, y):
        """
        
        Arguments:
        - `x`:
        - `y`:
        """

        val = x.dot(y)
        return val

class LapRLSClassifier(Classifier):
    """
    Laplacian Regularized Least Square Bninary Classifier.

    This class JUST coordinates binary classifiers for handling muti-classes.
    """

    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)
    logger = logging.getLogger("LapRLSClassifier")

    def __init__(self,
                 multi_class=model.MULTI_CLASS_ONE_VS_ONE,
                 lam=1,
                 normalized=True,
                 gamma_s=1,
                 kernel=model.KERNEL_RBF,
                 ):
        """
        """

        super(LapRLSClassifier, self).__init__(
            multi_class=multi_class,
        )

        self.lam = lam
        self.gamma_s = gamma_s
        self.normalized = normalized
        self.kernel_name = kernel

        self.logger.debug(
            """
            Parameters set with
            lambda = %f, kernel = %s, sigma = %f, multi_class is %s""" %
            (self.lam, self.kernel_name, self.gamma_s, self.multi_class))


    def create_binary_classifier(self, ):
        """
        """
        binary_classifier = LapRLSBinaryClassifier(
            lam=self.lam,
            gamma_s=self.gamma_s
        )
        return binary_classifier

    def _create_classifiers(self, param_grid=[{}]):
        """
        
        Arguments:
        - `param_grid`:
        """
        classifiers = []
        for param in param_grid:

            lam = param["lam"]
            gamma_s = param["gamma_s"]
            normalized = param["normalized"]
            kernel = param["kernel"]
            multi_class = param["multi_class"]

            classifier = LapRLSClassifier(
                multi_class=multi_class,
                lam=lam,
                normalized=normalized,
                gamma_s=gamma_s,
                kernel=kernel
            )
            classifiers.append(classifier)
        
        return classifiers
        
        
def main():

    # data
    data_path = "/home/kzk/datasets/uci_csv/glass.csv"
    data = np.loadtxt(data_path, delimiter=" ")
    y = data[:, 0]
    X = data[:, 1:]
    n = X.shape[0]
    X = np.hstack((X, np.reshape(np.ones(n), (n, 1))))
    X_l = X
    X_u = X

    # learn
    lam = 100
    gamma_s = .001
    model = LapRLSClassifier(lam=lam, normalized=False,
                             kernel="rbf",
                             gamma_s=gamma_s, multi_class="ovo")
    model.learn(X_l, y, X_u)

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
