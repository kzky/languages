#!/usr/bin/env python

"""
Confirmation for the recursion of rank-one update
"""

import numpy as np

# sample of  a matrix
alpha = 0.15
X = np.random.rand(100, 5)
I = np.diag(np.ones(X.shape[1]))
A = I + alpha * X.T.dot(X)
A_inv = np.linalg.inv(A)

print A_inv

# recursion update
B = I  # corresponding to invserse of I
for x in X:
    B = B - (alpha * B.dot(np.outer(x, x)).dot(B)) / (1 + alpha * x.dot(B).dot(x))
    pass

print B

d = A.shape[1]
for i in xrange(0, B.shape[0]):
    a = map(round, A_inv[i, :], [5] * d)
    b = map(round, B[i, :], [5] * d)
    print a==b
    pass



