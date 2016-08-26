import numpy as np
cimport numpy as np
from math import exp
from libc.math cimport exp as c_exp

def array_f(X):
    Y = np.zeros(X.shape)
    idx = np.where(X > 0.5)[0]
    Y[idx] = np.exp(X[idx])

    return Y

def c_array_f(np.ndarray X):

    cdef int N = X.shape[0]
    cdef double[:] Y = np.zeros(N)
    cdef int i

    for i in range(N):
        if X[i] > 0.5:
            Y[i] = c_exp(X[i])
        else:
            Y[i] = 0

    return Y
