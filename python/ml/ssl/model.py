#!/usr/bin/env python

import numpy as np

LEARN_TYPE_BATCH = "batch"
LEARN_TYPE_ONLINE = "online"

MULTI_CLASS_ONE_VS_ONE = "ovo"
MULTI_CLASS_ONE_VS_REST = "ovr"

class BinaryClassifier(object):
    """
    """
    
    def __init__(self, ):
        """
        """
        self.X_l = None
        self.y = None
        self.X_u = None
        self.X = None

        self.d = None
        self.u = None
        self.l = None

        self.I = None
        
        pass
        

    def learn(self, X_l, y, X_u):
        """
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """
        
        pass

    def predict(self, x):
        """
        
        Arguments:
        - `x`: sample, 1-d numpy array
        """
        
        pass

    def _normalize(self, X_l, X_u):
        """
        Nomalize dataset using some methods,
        e.g., zero mean, unit variance.
        """
        
        pass


    def _init_model(self, X_l, y, X_u):
        """
        Initialize model.
        No need to add bias term into samples because here bias is added.

        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array, y is in {-1, 1}
        - `X_u`: unlabeled samples, 2-d numpy array
        """
        # set the number of labeled samples, unlabeled samples, and dimension.
        shape_l = X_l.shape
        self.d = shape_l[1] + 1
        self.l = shape_l[0]

        shape_u = X_u.shape
        self.u = shape_u[0]
        self.n = self.l + self.u
        
        # set dataset and add bias
        self.X_l = np.hstack((X_l, np.reshape(np.ones(self.l), (self.l, 1))))
        self._check_and_set_y(y)
        self.X_u = np.hstack((X_u, np.reshape(np.ones(self.u), (self.u, 1))))
        self.X = np.vstack((self.X_l, self.X_u))

        # diagoanl matrix
        self.I = np.diag(np.ones(self.d))

        pass

    def _check_and_set_y(self, y):
        """
        Set y with checking for y definition
        """

        if not ((1 in y) and (-1 in y)):
            raise Exception("one which is not 1 or -1 is included in y")

        self.y = np.asarray(y)
        pass


class Classifier(object):
    """
    
    """
    
    def __init__(self, ):
        """
        """
        
        
        pass
        
    def learn(self, X_l, y, X_u):
        """
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """
        
        pass

    def predict(self, x):
        """
        
        Arguments:
        - `x`: sample, 1-d numpy array
        """
        
        pass
