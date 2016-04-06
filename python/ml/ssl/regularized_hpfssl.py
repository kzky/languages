#!/usr/bin/env python

from hpfssl import HPFSSLBinaryClassifier
from hpfssl import HPFSSLClassifier
from scipy import sparse

import scipy as sp
import numpy as np
import logging
import model

class RegularizedHPFSSLBinaryClassifier(HPFSSLBinaryClassifier):
    """
    This class is only for the batch mode.
    """
    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)
    logger = logging.getLogger("RegularizedHPFSSLBinaryClassifier")

    def __init__(self, max_itr=100, threshold=1e-4,
                 learn_type=model.LEARN_TYPE_ONLINE,
                 ):
        """
        Arguments:
        - `max_itr`: max iteration for stopping criterion
        - `threshold`: threshold for stopping criterion
        """

        super(RegularizedHPFSSLBinaryClassifier, self).__init__(
            max_itr=max_itr, threshold=threshold,
            learn_type=learn_type)

    def _compute_S_batch(self, esp=1e-12):
        """
        Compute covariance matrix
        
        """

        beta = self.beta
        XLX = self.XLX
        X_lX_l = self.X_lX_l
        small_I = esp*self.I

        S = np.linalg.inv(small_I + XLX + beta * X_lX_l)

        return S

class RegularizedHPFSSLClassifier(HPFSSLClassifier):
    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)
    logger = logging.getLogger("RegularizedHPFSSLClassifier")

    def __init__(self,
                 multi_class=model.MULTI_CLASS_ONE_VS_ONE,
                 max_itr=100, threshold=1e-4,
                 learn_type=model.LEARN_TYPE_ONLINE,
                 ):

        super(RegularizedHPFSSLClassifier, self).__init__(
            multi_class=multi_class,
            max_itr=max_itr,
            threshold=threshold,
            learn_type=learn_type
        )
        pass

    def create_binary_classifier(self, ):
        """
        """
        binary_classifier = RegularizedHPFSSLBinaryClassifier(
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

            classifier = RegularizedHPFSSLClassifier(
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
    l_data_path = "/home/kzk/datasets/uci_csv_ssl_lrate_fixed_1_50_1_98/car/4_l.csv"
    data_l = np.loadtxt(l_data_path, delimiter=" ")
    data_l = np.hstack((data_l, np.reshape(np.ones(data_l.shape[0]), (data_l.shape[0], 1))))
    y_l = data_l[:, 0]
    X_l = data_l[:, 1:]

    # unlabeled sample
    u_data_path = "/home/kzk/datasets/uci_csv_ssl_lrate_fixed_1_50_1_98/car/4_u.csv"
    data_u = np.loadtxt(u_data_path, delimiter=" ")
    data_u = np.hstack((data_u, np.reshape(np.ones(data_u.shape[0]), (data_u.shape[0], 1))))
    X_u = data_u[:, 1:]

    # test sample
    t_data_path = "/home/kzk/datasets/uci_csv_ssl_lrate_fixed_1_50_1_98/car/4_t.csv"
    data_t = np.loadtxt(t_data_path, delimiter=" ")
    data_t = np.hstack((data_t, np.reshape(np.ones(data_t.shape[0]), (data_t.shape[0], 1))))
    y_t = data_t[:, 0]
    X_t = data_t[:, 1:]
    
    # learn
    st = time.time()
    model = RegularizedHPFSSLClassifier(
        max_itr=10, threshold=1e-4,
        learn_type="online", multi_class="ovo")
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
    model = RegularizedHPFSSLClassifier(max_itr=0, threshold=1e-4, learn_type="online", multi_class="ovo")
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
    
