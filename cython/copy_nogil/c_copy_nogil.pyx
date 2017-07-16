import cython
from libc.string cimport memcpy

import numpy as np
cimport numpy as np

@cython.boundscheck(False)
def copy_array_nogil(double[:, :, :, :] x, double[:, :, :, :] y):
    cdef int size
    cdef int a, b, c, d
    cdef int i, j, k, l

    a = x.shape[0]
    b = x.shape[1]
    c = x.shape[2]
    d = x.shape[3]
    size = x.size * 8  # float64    

    with nogil:
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    for l in range(d):
                        x[i, j, k, l] = y[i, j, k, l]


