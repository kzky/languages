import numpy as np
import cython
cimport numpy as np
from math import exp
from libc.math cimport exp as c_exp
from cython.parallel import prange

def array_f(X):
    Y = np.zeros(X.shape)
    idx = np.where(X > 0.5)[0]
    Y[idx] = np.exp(X[idx])

    return Y

@cython.boundscheck(False) 
cdef inline void c_array_f_nogil(double[:] X, double[:] Y) nogil:

    cdef unsigned int N = X.shape[0]
    cdef unsigned int i
    
    for i in range(N):
        if X[i] > 0.5:
            Y[i] = c_exp(X[i])
        else:
            Y[i] = 0


def c_array_f(double[:] X):

    cdef double[:] Y = np.zeros(X.shape[0])
    with nogil:
        c_array_f_nogil(X, Y)

    print(np.asarray(Y))
    return Y
        
