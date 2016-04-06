#!/usr/bin/env python

import numpy as np
import time

from numba import jit

# dataset
n = 10000
d = 1000
X_ = np.random.rand(n, d)
y_ = np.random.rand(d)
b_ = np.random.rand(n)
gamma_ = 0.01
max_itr = 1000

# numpy
st0 = time.time()
for i in xrange(0, max_itr):
    X_.dot(y_) * gamma_ + b_
    pass
et0 = time.time()
print "elapsed time (numpy): %f [s]" % (et0 - st0)


# numba
@jit
def affine(X, y, gamma, b):
    """
    """
    n = X.shape[0]
    d = X.shape[1]

    # TOO BAD so to stick to direct manipulation
    #for i in xrange(0, max_itr):
    #    X_.dot(y_) * gamma_ + b_
    #    pass

    for k in xrange(0, max_itr):
        for i in range(n):
            for j in range(d):
                X[i, j] * y[j] * gamma_ + b_[i]
                pass
            pass
            
st0 = time.time()
affine(X_, y_, gamma_, b_)
et0 = time.time()
print "elapsed time (numba): %f [s]" % (et0 - st0)

# numba (type-specified)
@jit("(f8[:, :], f8[:], f8, f8[:])")
def affine_type_specified(X, y, gamma, b):
    """
    """
    n = X.shape[0]
    d = X.shape[1]

    # TOO BAD so to stick to direct manipulation
    #for i in xrange(0, max_itr):
    #    X_.dot(y_) * gamma_ + b_
    #    pass
    
    for k in xrange(0, max_itr):
        for i in range(n):
            for j in range(d):
                X[i, j] * y[j] * gamma_ + b_[i]
                pass
            pass

    
st0 = time.time()
affine_type_specified(X_, y_, gamma_, b_)
et0 = time.time()
print "elapsed time (numba type-specified): %f [s]" % (et0 - st0)

