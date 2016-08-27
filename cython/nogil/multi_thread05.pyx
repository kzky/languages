import numpy as np
import cython
cimport numpy as np
from math import exp
from libc.math cimport exp as c_exp
from cython.parallel import prange
from threading import Thread
from libcpp.vector cimport vector

def array_f(X):
    Y = np.zeros(X.shape)
    idx = np.where(X > 0.5)[0]
    Y[idx] = np.exp(X[idx])

    return Y

class TaskThread(Thread):

    def __init__(self, x):
        super(TaskThread, self).__init__()

        self.x = x
        self.y = None
        
    def run(self, ):
        y = self.c_array_f(self.x)
        self.y = np.asarray(y)
     
    def c_array_f(self, double[:] X):
     
        with nogil:
            Y = c_array_f_nogil(X)
     
        return Y
            
@cython.boundscheck(False) 
cdef inline vector[double] c_array_f_nogil(double[:] X) nogil:
 
    cdef unsigned int N = X.shape[0]
    cdef unsigned int i
    cdef vector[double] Y
    
    for i in range(N):
        if X[i] > 0.5:
            Y.push_back(c_exp(X[i]))
        else:
            Y.push_back(0)

    return Y
