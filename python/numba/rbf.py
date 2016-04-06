#!/usr/bin/env python

import numpy as np
import time

from numba import jit

# dataset
n = 1000
d = 10  # or 10
X = np.random.rand(n, d)
gamma = 0.01


# numpy
st = time.time()
for x in X:
    for y in X:
        np.exp(-gamma * np.sum((x - y) ** 2))
        pass
    pass
et = time.time()
print "elapsed time (numpy): %f [s]" % (et - st)
        
# numpy efficiently computed
def distmat(X):
    H = np.tile(np.diag(np.dot(X, X.T)), (X.shape[0], 1))
    G = np.dot(X, X.T)
    distmat_ = H - 2 * G + H.T
    return distmat_
st = time.time()
Y = np.exp(-gamma * distmat(X))
et = time.time()
print "elapsed time (numpy efficiently computed): %f [s]" % (et - st)


# for-loop with jit
@jit("f8[:, :](f8[:, :])")
def distmat_jit(X):
    """
    """
    Y = np.zeros((n, n))
    for i in range(n):
        x = X[i, :]
        for j in range(n):
            s = 0
            y = X[j, :]

            if i > j:
                Y[i, j] = Y[j, i]
                continue
                pass
            
            for k in range(d):
                s += (x[k] - y[k]) ** 2
                pass
            
            Y[i, j] = np.exp(-gamma * s)
            pass
    return Y
    
st = time.time()
distmat_jit(X)
et = time.time()
print "elapsed time (numba): %f [s]" % (et - st)
