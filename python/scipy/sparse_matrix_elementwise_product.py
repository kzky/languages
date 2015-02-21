#!/usr/bin/env python

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

# Sparse marrix element-wise product
## once creating diagonal matrix, then use "*", matrix-product
## http://stackoverflow.com/questions/3247775/how-to-elementwise-multiply-a-scipy-sparse-matrix-by-a-broadcasted-dense-1d-arra
X = np.random.rand(10)
X[2] = 0
X[7] = 0
X[8] = 0
X[9] = 0
X = csr_matrix(X)
f_dims = X.shape[1]
Z = lil_matrix((f_dims, f_dims))
Z.setdiag(X.toarray()[0])
print Z * X.transpose()

## todia
X = np.random.rand(10)
X[2] = 0
X[7] = 0
X[8] = 0
X[9] = 0
X = csr_matrix(X)
Z = X.todia()
print X.transpose() * Z
#  result is 1-by-1 matrix...


## DIY-element-wise product for 1-by-n sparse matrix
X = np.random.rand(10)
X[2] = 0
X[7] = 0
X[8] = 0
X[9] = 0
X = csr_matrix(X)

Y = np.random.rand(15)
Y[0] = 0
Y[2] = 0
Y[5] = 0
Y[8] = 0
Y = csr_matrix(Y)

indices = np.intersect1d(X.indices, Y.indices)
indptr = np.asarray([0, len(indices)], dtype="int32")
data = X.data[np.in1d(X.indices, indices)] * Y.data[np.in1d(Y.indices, indices)]

Z = csr_matrix((data, indices, indptr), shape=Y.shape)



