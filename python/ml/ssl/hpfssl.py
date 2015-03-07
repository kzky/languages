#!/usr/bin/env python

from model import Model
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import numpy as np
import logging

LEARN_TYPE_BATCH = "batch"
LEARN_TYPE_ONLINE = "online"

MULTI_CLASS_ONE_VS_ONE = "ovo"
MULTI_CLASS_ONE_VS_REST = "ovr"

class HPFSSLBinaryClassifier(Model):
    """
    Hyper Parameter Free Semi-Supervised Learning Binary Classifier.

    Original Binary Classifier, which update rule is as follows,
    
    m = \beta S X_l t
    S = (XLX^T + \beta X_l X_l^T) ^ {-1}
    L = X^{+} (S + mm^T)^{-1} X^T^{+})
    XLX^T = (S + mm^T)^{-1}
    \beta^{-1} = ( ||t - X_l^T m||_2^2 + \beta_{old}^{-1} Tr( I - SXLX^T) ) / l

    Notation is as follows,

    m: mean vector
    S: covariance matrix
    L: graph matrix
    beta: variance
    l: the number of labeled samples
    u: the number of unlabeled samples
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

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("HPFSSLBinaryClassifier")
    
    def __init__(self, max_itr=100, threshold=1e-6, learn_type="batch"):
        """
        Arguments:
        - `max_itr`: max iteration for stopping criterion
        - `threshold`: threshold for stopping criterion
        """

        super(HPFSSLBinaryClassifier, self).__init__()
        
        
        self.max_itr = max_itr
        self.threshold = threshold
        self.learn_type = learn_type

        self.X_l = None
        self.y = None
        self.X_u = None
        self.X = None
        self.m = None
        self.S = None
        self.XLX = None
        self.beta = None

        if learn_type == LEARN_TYPE_BATCH:
            self.learn = self._learn_batch
        elif learn_type == LEARN_TYPE_ONLINE:
            self.learn = self._learn_online
        else:
            raise Exception("learn_type %s does not exist." % learn_type)

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

    def _learn_batch(self, X_l, y, X_u):
        """
        Learn in the batch fashion
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """
        # init model
        self._init_model(X_l, y, X_u)
        self._compute_rank_one_sum()
        
        # initialize L, beta, S, and m
        self.XLX = self.I
        self.beta = 1
        self.S = self._compute_S_batch()
        self.m = self._compute_m_batch()
        
        t = 0
        while (t < self.max_itr):
            t += 1
            # update parameters
            self.XLX = self._compute_XLX_batch()
            self.beta = self._compute_beta_batch()
            self.S = self._compute_S_batch()
            m_old = self.m
            self.m = self._compute_m_batch()

            # check stopping criteria
            if t % 100 == 0:
                self.logger.info("itr: %d" % t)
                pass
            if self._check_stopping_criteria_with_m(m_old):
                break

        pass

    # TODO see more carefully
    def _learn_online(self, X_l, y, X_u):
        """
        Learn in the online fashion
        
        Arguments:
        - `X_l`: samples, 2-d numpy array
        - `y`: labels, 1-d numpy array
        - `X_u`: unlabeled samples, 2-d numpy array
        """

        # init model
        self._init_model(X_l, y, X_u)
        self._compute_rank_one_sum()
        
        # initialize L, beta, S, and m
        self.XLX = self.I
        self.beta = 1
        self.S = self._compute_initial_S_online()
        self.m = self._compute_m_online()

        t = 0
        while (t < self.max_itr):  # outer loop
            t += 1
            
            # update parameters
            self.XLX = self._compute_XLX_online()
            self.beta = self._compute_beta_online()
            m_old = self.m
            self.S = self._compute_S_online()
            self.m = self._compute_m_online()

            # check stopping criteria
            if t % 100 == 0:
                self.logger.info("itr: %d" % t)
                pass
            if self._check_stopping_criteria_with_m(m_old):
                break
        pass

    def _check_stopping_criteria_with_m(self, m_old):
        """
        
        Arguments:
        - `m_old`:
        """
        d = self.m - m_old
        d_L2_norm = np.sqrt(np.sum(d**2))
        if d_L2_norm < self.threshold:
            return True
        else:
            return False

        pass

    def _compute_rank_one_sum(self, ):
        """
        Compute rank one sum.
        """
        self.X_lX_l = self.X_l.T.dot(self.X_l)

    def _compute_m_batch(self,):
        """
        Compute mean vector.
        """

        X_l = self.X_l
        y = self.y
        beta = self.beta
        S = self.S
        
        m = beta * S.dot(X_l.T.dot(y))
        return m

    def _compute_S_batch(self, ):
        """
        Compute covariance matrix
        
        """

        beta = self.beta
        XLX = self.XLX
        X_lX_l = self.X_lX_l

        S = np.linalg.inv(XLX + beta * X_lX_l)
        return S

    def _compute_XLX_batch(self,):
        """
        Compute L

        """
        m = self.m
        S = self.S

        # (n-by-d)-matrix
        XLX = np.linalg.inv(S + np.outer(m, m))

        return XLX

    def _compute_beta_batch(self, ):
        """
        Compute beta
        
        Arguments:
        - `m`:
        - `S`:
        - `beta_old`:
        """

        X_l = self.X_l
        y = self.y
        I = self.I
        m = self.m
        S = self.S
        XLX = self.XLX
        beta_old = self.beta
        l = self.l
        
        residual = y - X_l.dot(m)
        inner_trace = I - S.dot(XLX)

        # Note that this is similar to RVM update rule and PSD holds.
        beta = l / (np.sum(residual ** 2) + inner_trace.trace() / beta_old)

        return beta

    def _compute_m_online(self, ):
        """
        Compute m in an online fashion
        
        Arguments:
        - `x`: sample, 1-d numpy array, where x is a labeled sample
        - `y`: label
        
        """
        
        return self._compute_m_batch()

    def _compute_initial_S_online(self, ):
        """
        Compute S in an online fashion
        """
        
        XLX_inv = self.I
        S_t = XLX_inv

        for x in self.X_l:
            S_t = S_t - S_t.dot(np.outer(x, x)).dot(S_t) / (1 + x.dot(S_t).dot(x))
            pass

        return S_t

    def _compute_S_online(self, ):
        """
        Compute S in an online fashion
        """

        S = self.S
        m = self.m
        XLX_inv = S + np.outer(m, m)

        S_t = XLX_inv
        beta = self.beta

        for x in self.X_l:
            S_t = S_t - beta * S_t.dot(np.outer(x, x)).dot(S_t) / (1 + beta * x.dot(S_t).dot(x))
            pass

        return S_t

    def _compute_XLX_online(self, ):
        """
        """

        XLX = self.XLX
        beta = self.beta
        X_lX_l = self.X_lX_l
        S_inv = XLX + beta * X_lX_l
        m = self.m

        XLX = S_inv - S_inv.dot(np.outer(m, m)).dot(S_inv) / (1 + m.dot(S_inv).dot(m))

        return XLX
        
    def _compute_beta_online(self, ):
        """
        """
        return self._compute_beta_batch()

        
    def predict(self, x):
        """
        Predict label for x
        
        Arguments:
        - `x`: sample, 1-d numpy array
        """

        x = np.hstack((x, 1))
        
        return np.sum(self.m * x)

class HPFSSLClassifier(Model):
    """
    HPFSSLClassifier handles multi-class with HPFSSLBinaryClassifier.

    This class JUST coordinates binary classifiers for handling muti-classes.
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("HPFSSLClassifier")
    
    def __init__(self, max_itr=100, threshold=1e-6, learn_type="batch", multi_class="ovo"):
        """
        """

        super(HPFSSLClassifier, self).__init__()

        # params
        self.max_itr = max_itr
        self.threshold = threshold
        self.learn_type = learn_type
        self.multi_class = multi_class

        # pairs
        self.pairs = list()

        # classes
        self.classes = list()
        
        # model
        self.models = dict()

        self.logger.info("learn_type is %s" % self.learn_type)
        self.logger.info("multi_class is %s" % self.multi_class)
        
        if multi_class == MULTI_CLASS_ONE_VS_ONE:
            self.learn = self._learn_ovo
            self.predict = self._predict_ovo
        elif multi_class == MULTI_CLASS_ONE_VS_REST:
            self.learn = self._learn_ovr
            self.predict = self._predict_ovr
            
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
            model = HPFSSLBinaryClassifier(max_itr=self.max_itr, threshold=self.threshold, learn_type=self.learn_type)
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
        for pair, model in self.models.items():
            target = model.predict(x)
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
            model = HPFSSLBinaryClassifier(max_itr=self.max_itr, threshold=self.threshold, learn_type=self.learn_type)
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
        
def main():

    # data
    data_path = "/home/kzk/datasets/uci_csv/glass.csv"
    data = np.loadtxt(data_path, delimiter=" ")
    y = data[:, 0]
    X = data[:, 1:]

    # learn
    model = HPFSSLClassifier(max_itr=100, threshold=1e-6, learn_type="online", multi_class="ovo")
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
