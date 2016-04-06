#!/usr/bin/env python

import numpy as np
import time

from scipy.spatial.distance import pdist

# dataset
n = 1000
d = 10
X = np.random.rand(n, d)
gamma = 0.01

# for-loop
st = time.time()
Y = np.zeros((n, n))
for i, x in enumerate(X):
    for j, y in enumerate(X):
        Y[i, j] = np.exp(-gamma * np.sum((x - y) ** 2))
    pass
et = time.time()
print "elapsed time (for-loop): %f [s]" % (et - st)

# func style
def distmat(X):
    H = np.tile(np.diag(np.dot(X, X.T)), (X.shape[0], 1))
    G = np.dot(X, X.T)
    DistMat = H - 2 * G + H.T
    return DistMat
st = time.time()
Y = np.exp(-gamma * distmat(X))
print Y.shape
et = time.time()
print "elapsed time (func style): %f [s]" % (et - st)

# scipy.spatial.distance.pdist (half matrix return)
st = time.time()
Y = pdist(X, 'euclidean')
Y = np.exp(-gamma * (Y ** 2))
et = time.time()
print "elapsed time (sicpy.spatial.distance.pdist): %f [s]" % (et - st)

# scipy.spatial.distance.pdist (with lambda) too slow as for-loop (half matrix return)
st = time.time()
Y = pdist(X, lambda u, v: np.exp(-gamma * np.sum((u - v) ** 2)))
et = time.time()
print "elapsed time (sicpy.spatial.distance.pdist with lambda): %f [s]" % (et - st)



