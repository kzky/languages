#!/usr/bin/env python

import numpy as np
import logging

class GridSearchValidator(object):
    """
    Grid search validator.
    """
    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)
    logger = logging.getLogger("GridSearchValidator")
    
    def __init__(self, classifiers=[]):
        """
        Arguments:
        - `classifiers`: classifier list

        """

        self.classifiers = classifiers
        pass

    def set_classifiers(self, classifiers=[]):
        """
        
        Arguments:
        - `classifiers`:
        """

        self.classifiers = classifiers
        

    def validate_in_ssl(self, X_l, y, X_u, X_v, y_v):
        """
        Return most confidential classifier index in terms of accuracy.
        
        """

        accs = []
        for i, classifier in enumerate(self.classifiers):
            classifier.learn(X_l, y, X_u)
            y_preds = np.asarray(classifier.predict_classes(X_v))
            n_hits = len(np.where(y_preds == y_v)[0])
            accs.append(n_hits)

        max_idx = np.argmax(accs)
        return self.classifiers[max_idx]


    def validate(self, X, y, X_v, y_v):
        """
        Return most confidential classifier index in terms of accuracy.
        
        """

        accs = []
        for i, classifier in enumerate(self.classifiers):
            classifier.learn(X, y)
            y_preds = np.asarray(classifier.predict_classes(X_v))
            n_hits = len(np.where(y_preds == y_v)[0])
            accs.append(n_hits)

        max_idx = np.argmax(accs)
        return self.classifiers[max_idx]
        
