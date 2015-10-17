#!/usr/bin/env python

"""
NOTE: This is NOT ssl version of SVM. For the sake of experiment, this model extis and aligns interface for SSL.
"""

from sklearn.svm import LinearSVC
from model import BinaryClassifier
from model import Classifier
from sklearn.metrics import confusion_matrix
import numpy as np
import logging
import model

class LSVMBinaryClassifier(BinaryClassifier):
    """
    Linear SVM BinaryClassifier wrapnig Liblinear of L2-Regularizer and
    L2-Hinge Loss.
    """
    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)
    logger = logging.getLogger("LSVMBinaryClassifier")
    
    def __init__(self, C=1):
        """
        
        Arguments:
        - `C`: Balancing cnostant between loss and regularizer
        """
        super(LSVMBinaryClassifier, self).__init__()
        
        self.C = C

        # intercept is always added into sample.
        self.model = LinearSVC(C=self.C, loss="l2", penalty="l2",
                               tol=1e-4, dual=False, fit_intercept=False)

        
    def learn(self, X_l, y, X_u):
        """
        
        Arguments:
        - `X_l`:
        - `y`:
        - `X_u`:
        """
        self._set_data_info(X_l, y, X_u)
        self._learn_batch(self.X_l, self.y, self.X_u)

    def _learn_batch(self, X_l, y, X_u):
        """
        
        Arguments:
        - `X_l`:
        - `y`:
        - `X_u`:
        """

        self.model.fit(X_l, y)

    def predict(self, x):
        """
        
        Arguments:
        - `x`: sample, 1-d numpy array
        """
        val = self.model.decision_function(x)[0]

        return val
            
class LSVMClassifier(Classifier):
    """
    Linear SVM Classifier wrapnig Liblinear of L2-Regularizer and L2-Hinge Loss.
    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("LSVMClassifier")

    def __init__(self, C=1, multi_class=model.MULTI_CLASS_ONE_VS_ONE):
        """
        """

        super(LSVMClassifier, self).__init__(multi_class=multi_class)

        self.C = C

        self.logger.debug("Parameters set with C=%f" % (self.C))

    def create_binary_classifier(self, ):
        """
        """
        binary_classifier = LSVMBinaryClassifier(C=self.C)

        return binary_classifier

    def _create_classifiers(self, param_grid=[{}]):
        """
        
        Arguments:
        - `param_grid`:
        """
        classifiers = []
        for param in param_grid:
            C = param["C"]
            
            classifier = LSVMClassifier(
                C=C,
            )
            classifiers.append(classifier)

        return classifiers

def main():

    # data
    data_path = "/home/kzk/datasets/uci_csv/spam.csv"
    data = np.loadtxt(data_path, delimiter=" ")
    y = data[:, 0]
    X = data[:, 1:]
    n = X.shape[0]
    X = np.hstack((X, np.reshape(np.ones(n), (n, 1))))
    X_l = X
    X_u = X

    # learn
    C = 1
    model = LSVMClassifier(multi_class="ovo", C=C)
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
