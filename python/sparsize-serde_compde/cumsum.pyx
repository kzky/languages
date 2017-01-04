from __future__ import division
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
cpdef cumsum(np.ndarray[np.int_t, ndim=1] x, unsigned int n):
    """Cumularive summation in the destructive manner
    """
    cdef unsigned int i = 1
    #cdef np.ndarray[np.int_t, ndim=1] x_ = np.zeros(n, dtype=np.int)
    #x_[0] = x[0]
    for i in range(n-1):
        x[i+1] = x[i] + x[i+1]
    
    
