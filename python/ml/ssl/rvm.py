#!/usr/bin/env python

"""
NOTE: This is NOT ssl version of RVM. For the sake of experiment, this model extis and aligns interface for SSL.
"""

from sklearn.metrics import confusion_matrix
from ml.sl import rvm
import numpy as np
import logging
import model

class RVMClassifier(object):
    """
    Linear RVM Classifier wrapnig Liblinear of L2-Regularizer and L2-Hinge Loss.
    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("RVMClassifier")

    def __init__(self,
                 multi_class=model.MULTI_CLASS_ONE_VS_ONE,
                 max_itr=100, threshold=1e-4,
                 learn_type=model.LEARN_TYPE_ONLINE,
                 alpha_threshold=1e-24,
                 ):
        """
        """

        self.wrapped_model = rvm.RVMClassifier(
            max_itr=max_itr,
            threshold=threshold,
            learn_type=learn_type,
            multi_class=multi_class,
            alpha_threshold=alpha_threshold)

    def learn(self, X_l, y, X_u):
        """
        
        Arguments:
        - `X_l`:
        - `y`:
        - `X_u`:
        """

        self.wrapped_model.learn(X_l, y)
        

    def predict(self, x):
        """
        
        Arguments:
        - `x`:
        """
        return self.wrapped_model.predict(x)

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
    model = RVMClassifier(multi_class="ovo")
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
