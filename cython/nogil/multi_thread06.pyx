import numpy as np
import cython
cimport numpy as np
from math import exp
from libc.math cimport exp as c_exp
from cython.parallel import prange
from threading import Thread
from cpython cimport array

def array_f(X):
    Y = np.zeros(X.shape)
    idx = np.where(X > 0.5)[0]
    return idx

class TaskThread(Thread):

    def __init__(self, x):
        super(TaskThread, self).__init__()

        self.x = x
        self.y = None
        
    def run(self, ):
        y = self.c_array_f(self.x)
        self.y = np.asarray(y)

    def c_array_f(self, double[:] X):
        idx= _c_array_f(X)
        return idx
    
cdef inline int[:] _c_array_f(double[:] X):
    cdef int c = 0
    with nogil:
        c = c_array_f_nogil(X)
    
    cdef int[:] idx = np.ndarray(c, dtype=np.int32)
    with nogil:
        c_array_f2_nogil(X, idx)
 
    return idx
            
@cython.boundscheck(False) 
cdef inline int c_array_f_nogil(double[:] X) nogil:
    cdef unsigned int N = X.shape[0]
    cdef unsigned int i
    cdef unsigned int c = 0
    
    for i in range(N):
        if X[i] > 0.5:
            c += 1
    return c
            
@cython.boundscheck(False) 
cdef inline void c_array_f2_nogil(double[:] X, int[:] idx) nogil:
 
    cdef unsigned int N = X.shape[0]
    cdef unsigned int i
    cdef unsigned int c = 0
    
    for i in range(N):
        if X[i] > 0.5:
            idx[c] = i
            c += 1

    
