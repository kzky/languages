import numpy as np
import scipy as sp
import logging as logger
import time
import kernels
from collections import defaultdict
from numpy.linalg import inv
from sklearn.metrics import confusion_matrix

class RDM(object):
    """
    Relevance Vector Machine.
    Type-2 Bayesian Estimation for Squared-Loss + Regularization
        
    References:
    - http://machinelearning.wustl.edu/mlpapers/paper_files/Tipping01.pdf
    - http://en.wikipedia.org/wiki/Relevance_vector_machine
    - http://www.amazon.co.jp/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738/ref=sr_1_1?s=english-books&ie=UTF8&qid=1400331080&sr=1-1&keywords=pattern+recognition+and+machine+learning
    """
    
    def __init__(self, fname, delimiter = " ", kernel = "rbf", itr = 100, epsilon = 10 ** -16):
        """
        init
        """
        logger.basicConfig(level=logger.DEBUG)
        logger.info("init starts")

        self.data = defaultdict()
        self.model = defaultdict()
        
        self._load(fname, delimiter)
        self._init_model(kernel, t, epsilon)

    def _load(self, fname, delimiter = " "):
        """
        Load data set specified with filename.

        Arguments:
        - `fname`:
        - `delimiter`:
        """
        # load data
        st = time.time()
        data = np.loadtxt(fname, delimiter = delimiter)
        et = time.time()
        logger.info("loading data time: %f[s]", (et - st))

        self.data["n_samples"] = data.shape[0] 
        self.data["f_dim"] = data.shape[1]# - 1 + 1
        self.data["classes"] = np.unique(data[:, 0])
        self.data["n_classes"] = len(self.data["classes"])

        # normlize
        self.normalize(data[:, 1:])

        biases = np.ones(self.data["n_samples"])
        self.data["data"] = np.column_stack((data, biases))

        logger.info("init finished")
        
    def _init_model(self, kernel = "rbf", itr = 100, epsilon = 10 ** -16):
        """
        Initialize model.
        Pair-waise Classification.
        """
        
        # set parameter
        self.model["itr"] = itr
        self.model["epsilon"] = epsilon
        
        # set kernel
        _set_kernel()

        # pairwase
        _pairwise()
        
    def _set_kernel(self, ):
        """
        set kernel
        """
        kernels = Kernels()
        if hasattr(kernels, kernel):
            attr = getattr(kernels, kernel)
            if callable(attr):
                self.model["kernel"] = attr()
            else:
                print "kernel", kernl, "does not exist in Kernels."
                exit(1)

    def _pairwise(self, ):
        """
        pairwise classes/samples
        """
        classes = self.data["classes"]
        self.model["pair_labels"] = defaultdict()
        self.data["pair_samples"] = defaultdict()
        
        # pairwise for classes/samples
        for i in xrange(self.data["n_classes"]):
            for j in xrange(self.data["n_classes"]):
                if i < j:
                    cls1 = classes[i]
                    cls2 = classes[j]
                    pair = str(cls1) + "," + str(cls2)
                    
                    # pairwise labels/binary labels
                    labels = _pairwise_labels(cls1, cls2)
                    binary_labels = _binalize(lables)
                    self.data["pair_labels"][pair] = defaultdict()
                    self.data["pair_labels"][pair]["labels"] = labels
                    self.data["pair_labels"][pair]["binary_labels"] = binary_labels
                    
                    # pairwise samples
                    self.data["pair_samples"][pair] = self._pairwise_samples(cls1, cls2)
                    
    def _pairwise_samples(self, cls1, cls2):
        """
        pairwise samples
        Arguments:
        - `cls1`: class 1
        - `cls2`: class 2
        """
        labels = self.data["data"][, :0]
        return self.data["data"][(labels==cls1) | (labels==cls2), 1:]

    def kernelize_pairwisely(self, cls1, cls2):
        """
        kernelize all samples
        Arguments:
        - `data`:
        """
        n_samples = self.data["data"]["n_samples"]
        k = self.model["kernel"]
        data = self.data["data"]
        labels = self.data["data"][, :0]
        samples = data[(lables==cls1)|(lables==cls2), 1:]# duplicate data but in local scope
        
        n_samples = len(samples)
        K = np.identity(n_samples)
        for i in xrange(n_samples):
            for j in xrange(n_samples):
                if i <= j:
                    K[i, j] = k(samples[i, :], samples[j, :])
                else:
                    K[j, i] = K[i, j]

        return K

    def _pairwise_labels(self, cls1, cls2):
        """
        pairwise labels
        `cls1`: class 1
        `cls2`: class 2
        """
        labels = self.data["data"][, :0]
        return labels[(lables == cls1) | (lables == cls2)]
                
    def _binalize(self, labels):
        """
        Binalize pairwise labels
        
        Arguments:
        - `lables`: pairwise labels
        """
        logger.info("binalize starts")
        
        # binary check
        classes = np.unique(labels)
        if classes.size != 2:
            print "label must be a binary value."
            exit(1)

        # map<binary value, class>
        self.data["pair_labels"][pair]["map"] = defaultdict()
        self.data["pair_labels"][pair]["map"][1] = classes[0]
        self.data["pair_labels"][pair]["map"][-1] = classes[1]

        # convert binary lables to {1, -1}
        _lables = np.zeros(labels.size)
        for i in xrange(labels.size):
            if binary_labels[i] == classes[0]: 
                binary_labels[i] = 1
            else:
                binary_labels[i] = -1

        logger.info("binalize finished")
        return binary_labels

    def normalize(self, samples):
        """
        nomalize sample, such that sqrt(x^2) = 1
        
        Arguments:
        - `samples`: dataset without labels.
        """
        logger.info("normalize starts")
        for i in xrange(0, self.data["n_sample"]):
            samples[i, :] = self._normalize(samples[i, :])
            
        logger.info("normalize finished")

    def _normalize(self, sample):
        norm = np.sqrt(sample.dot(sample))
        sample = sample/norm
        return sample

    def _add_bias(self, sample):
        return np.hstack((sample, 1))

    def learn(self, ):
        """
        Learn RVM for all class
        """
        # pair-wise classifier
        for i in xrange(self.data["n_classes"]):
            for j in xrange(self.data["n_classes"]):
                if i < j:
                    cls1 = classes[i]
                    cls2 = classes[j]
                    pair = str(cls1) + "," + str(cls2)

                    # kernelize pairwisely
                    _learn(pair)
                    
    def _learn(self, cls1, cls2):
        """
        Learn RVM for one model
        Arguments:
        - `pair`: class pair
        """
        pair = str(cls1) + "," + str(cls2)
        itr = self.model["itr"]
        _itr = 0
        t = self.data["pair_labels"][pair]["binary_labels"]
        K = _kernelize_pairwisely(cls1, cls2)
        Kt = K.dot(t)
        KK = K.dot(K) # this implementation for the kernlized version only.

        # init model parameters
        alpha = np.ones(K.shape[0])
        beta = 1
        S = inv(np.diag(alpha) + beta*KK)
        m = beta * S.dot(Kt)
        ones = np.ones(t.size)

        while (itr < threshold):
            gamma = ones - alpha * np.diag(S)
            alpha = gamma/(m ** 2)
            de = t - K.dot(m)
            beta = (l - np.sum(gamma)) / (de.dot(de))
            S = np.linalg.inv(np.diag(alpha) + beta * KK)
            m = beta * S.dot(Kt)
            _itr += 1

            # TODO 
            # add stopping criterion for eplison

        # save
        self.model["params"][pair]["m"] = m
        ## those should be in memory?
        self.model["params"][pair] = defaultdict()
        self.model["params"][pair]["beta"] = beta
        self.model["params"][pair]["S"] = S 

    def predict(self, samples):
        """
        Predict for all samples.
        Arguments:
        - `samples`:
        """
        n_samples = samples.shape[0]
        outputs = nd.array(n_samples)
        for i in xrange(n_samples):
            outputs[i] = _predict(samples[i, :])

        return outputs
        
    def _predict(self, sample):
        """
        Predict for one sample.
        Arguments:
        - `sample`:
        """
        # pair-wise classifier
        n_classifiers = len(self.model["params"])
        outputs = defaultdict()
        outputs.default_factory = int
        labels_map = self.data["pair_labels"][pair]["map"]
        sample = self._add_bias(sample)

        for i in xrange(self.data["n_classes"]):
            for j in xrange(self.data["n_classes"]):
                if i < j:
                    cls1 = classes[i]
                    cls2 = classes[j]
                    pair = str(cls1) + "," + str(cls2)
                    m = self.model["params"][pair]["m"]
                    kernel = self.model["kernel"]
                    samples = self.data["pair_samples"][pair]
                    n_samples = samples.shape[0]


                    sum = 0
                    for k in xrange(n_samples):
                        sum += kernel(sample, samples[k, :])

                    # voting
                    if sum >= 1:
                        outputs[lables_map[1]] += 1
                    else:
                        outputs[lables_map[-1]] += 1
                        
        return max(outputs, outputs.get)

    @classmethod
    def examplify(cls, fname, delimiter = " ", kernel = "rbf", itr = 100, epsilon = 10 ** -16):
        """
        Example of how to use
        """
        
        # learn
        model = RVM(fname, delimiter = delimiter, kernel = kernel, itr = itr, epsilon = epsilon)

        model.learn()

        # load data
        data = np.loadtxt(fname, delimiter = delimiter)

        # normalize all samples
        model._normalize(data[:, 1:])

        # predict
        y_pred = model.predict(data[:, 1:])
        
        # show result
        y_label = data[:, 0]
        cm = confusion_matrix(y_label, y_pred)
        print cm
        print "accurary: %d [%%]" % (np.sum(cm.diagonal()) * 100.0/np.sum(cm))

if __name__ == '__main__':
    fname = "/home/kzk/datasets/uci_csv/liver.csv"
    #fname = "/home/kzk/datasets/uci_csv/ad.csv"
    #fname = "/home/kzk/datasets/uci_csv/adult.csv"
    #fname = "/home/kzk/datasets/uci_csv/iris2.csv"
    print "dataset is", fname

    # TODO
    # test
    RVM.examplify(fname, delimiter = " ", kernel = "rbf", itr = 100, epsilon = 10 ** -16)

