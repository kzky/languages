#!/usr/bin/env python

import numpy as np

# dataset
X_ = np.random.rand(20, 10)
gamma_ = 0.01

Y_ = np.reshape(X_, (X_.shape[0], 1, X_.shape[0]))
print Y_.shape


