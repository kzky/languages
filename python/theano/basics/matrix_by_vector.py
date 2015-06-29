#!/usr/bin/env python

import numpy as np
import theano.tensor as T
import time

from theano import function

# dataset
n = 10000
d = 10000
X_ = np.random.rand(n, d)
v_ = np.random.rand(d, 1)
max_itr = 500

# numpy
st0 = time.time()
for i in xrange(0, max_itr):
    X_.dot(v_)
    pass
et0 = time.time()
print "elapsed time (Numpy): %f [s]" % (et0 - st0)

# theano
X = T.dmatrix("X")
v = T.dmatrix("v")
Xv = function([X, v], T.dot(X, v))
st1 = time.time()
for i in xrange(0, max_itr):
    Xv(X_, v_)
    pass

et1 = time.time()
print "elapsed time (Theano): %f [s]" % (et1 - st1)
