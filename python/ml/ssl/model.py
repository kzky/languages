#!/usr/bin/env python

from collections import defaultdict
import numpy as np


LEARN_TYPE_BATCH = "batch"
LEARN_TYPE_ONLINE = "online"

MULTI_CLASS_ONE_VS_ONE = "ovo"
MULTI_CLASS_ONE_VS_REST = "ovr"

# TODO refacotr for arguments
class BinaryClassifier(object):
    """
    """
    
    def __init__(self,
                 max_itr=100, threshold=1e-6,
                 learn_type=LEARN_TYPE_BATCH,
                 lam=1, sigma_s=1,
                 **kwargs):

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

        
# TODO refacotr for arguments
class Classifier(object):
    """
    
    """
    def __init__(self, multi_class=MULTI_CLASS_ONE_VS_ONE,
                 max_itr=100, threshold=1e-6,
                 learn_type=LEARN_TYPE_ONLINE,
                 lam=1,
                 sigma_s=1,
                 **kwargs
                 ):
        """
        """

        # params
        self.multi_class = multi_class
        self.max_itr = max_itr
        self.threshold = threshold
        self.learn_type = learn_type
        self.lam = lam
        self.sigma_s = sigma_s

        # pairs
        self.pairs = list()

        # classes
        self.classes = list()

        # model
        self.models = dict()
        self.internal_classifier = BinaryClassifier

        
        if multi_class == MULTI_CLASS_ONE_VS_ONE:
            self.learn = self._learn_ovo
            self.predict = self._predict_ovo
        elif multi_class == MULTI_CLASS_ONE_VS_REST:
            self.learn = self._learn_ovr
            self.predict = self._predict_ovr
        else:
            raise Exception("multi_class is set with %s or %s" %
                            (MULTI_CLASS_ONE_VS_ONE, MULTI_CLASS_ONE_VS_REST))

    def _learn_ovo(self, X_l, y, X_u):
        """
        Learn with One-vs-One scheme for multiclass
        """

        # take set and sort
        classes = list(set(y))
        classes.sort()
        self.classes = classes
        
        # create pairs
        for i, c in enumerate(classes):
            for j, k in enumerate(classes):
                if i < j:
                    self.pairs.append((c, k))
                    pass

        # for each pair
        y = np.asarray(y)
        for pair in self.pairs:
            self.logger.info("processing class-pair (%s, %s)" % (pair[0], pair[1]))
            # retrieve indices
            idx_1 = np.where(y == pair[0])[0]
            idx_1_1 = np.where(y == pair[1])[0]

            # get samples X for a pair
            X_1 = X_l[idx_1, :]
            X_1_1 = X_l[idx_1_1, :]
            X_l_pair = np.vstack((X_1, X_1_1))
            
            # create y in {1, -1} corresponding to (c_i, c_{i+1})
            y_1 = [1] * len(idx_1)
            y_1_1 = [-1] * len(idx_1_1)
            y_pair = y_1 + y_1_1
            
            # pass (X_l, y, X_u) to binary classifier
            model = self.internal_classifier(max_itr=self.max_itr,
                                             threshold=self.threshold, learn_type=self.learn_type,
                                             lam=self.lam,
                                             sigma_s=self.sigma_s)
            model.learn(X_l_pair, y_pair, X_u)
            self.models[pair] = model
            
        pass

    def _predict_ovo(self, x):
        """
        Format of return is as follows sorted by values in descending order,

        [(c_i, v_i), (c_i, v_j), ...],

        where
        c_i is a class,
        v_i is a predicted values corresponding to c_i.

        Arguments:
        - `x`: sample, 1-d numpy array
        """
        votes = defaultdict(int)
        for pair, model_ in self.models.items():
            target = model_.predict(x)
            if target >= 0:
                votes[pair[0]] += 1
            else:
                votes[pair[1]] += 1

        outputs = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        return outputs
        
    def _learn_ovr(self, X_l, y, X_u):
        """
        Learn with One-vs-Rest scheme for multiclass
        """
        # take set and sort
        classes = list(set(y))
        classes.sort()
        self.classes = classes
        
        # for class
        y = np.asarray(y)
        for c in classes:
            self.logger.info("processing class %s" % c)
            # retrieve indices
            idx_1 = np.where(y == c)[0]
            idx_1_1 = np.where(y != c)[0]
            
            # get samples X for a pair
            X_1 = X_l[idx_1, :]
            X_1_1 = X_l[idx_1_1, :]
            X_l_pair = np.vstack((X_1, X_1_1))
            
            # create y in {1, -1} corresponding to (c_i, c_{i+1})
            y_1 = [1] * len(idx_1)
            y_1_1 = [-1] * len(idx_1_1)
            y_pair = y_1 + y_1_1

            # pass (X_l, y, X_u) to binary classifier
            model = self.internal_classifier(max_itr=self.max_itr,
                                             threshold=self.threshold, learn_type=self.learn_type)
            model.learn(X_l_pair, y_pair, X_u)
            self.models[c] = model
            
        pass

    def _predict_ovr(self, x):
        """
        Format of return is as follows sorted by values in descending order,

        [(c_i, v_i), (c_i, v_j), ...],

        where
        c_i is a class,
        v_i is a predicted values corresponding to c_i.

        Arguments:
        - `x`: sample, 1-d numpy array
        """
        outputs = defaultdict(int)
        for c, model in self.models.items():
            outputs[c] = model.predict(x)
            
        return sorted(outputs.items(), key=lambda x: x[1], reverse=True)
