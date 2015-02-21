#!/usr/bin/env python

import numpy as np
import theano.tensor as T
from theano import function
from theano import shared

print "Acturally, subgradient in theano is not subgradient"
print "In definition of subgradient, subgradient is a set of gradient"
print "By definitoin, the subgradient of f at x_0 is f(x) - f(x_0) >= v^T (x - x_0)"
print "e.g., subgradient of |x| at 0 = [-1 , 1], otherwise -1 if x > 0 or 1 if x <0"

print "Subgradient of L1-norm"
w = T.dvector("w")
w_norm_1 = w.norm(L=1)
gw = T.grad(w_norm_1, w)
f = function([w], gw)
w_ = np.random.normal(0, 1, 5)
w_[3] = 0
print f(w_)

print "Subgradient of max(x)"
x = T.dvector("x")
max_x = T.max(x)
gx = T.grad(max_x, x)
f = function([x], gx)
x_ = np.random.normal(0, 1, 5)
x_[3] = 0
print f(x_)



