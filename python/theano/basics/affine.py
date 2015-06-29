#!/usr/bin/env python

import numpy as np
import theano.tensor as T
import time

from theano import function
from theano import shared

# dataset
n = 10000
d = 1000
X_ = np.random.rand(n, d)
y_ = np.random.rand(d)
b_ = np.random.rand(n)
gamma_ = 0.01
max_itr = 100000

# numpy
st0 = time.time()
for i in xrange(0, max_itr):
    X_.dot(y_) * gamma_ + b_
    pass
et0 = time.time()
print "elapsed time (numpy): %f [s]" % (et0 - st0)

# theano
X = shared(X_)
y = T.dvector("y")
b = shared(b_)
gamma = T.dscalar("gamma")
Xyg_b = function([y, gamma], gamma * X.dot(y) + b)
st1 = time.time()
for i in xrange(0, max_itr):
    Xyg_b(y_, gamma_)
    pass

et1 = time.time()
print "elapsed time (Theano): %f [s]" % (et1 - st1)

"""
max_itr = 100000
Using gpu device 0: GeForce GTX 780
elapsed time (numpy): 703.786513 [s]
elapsed time (theano): 574.140746 [s]
"""
