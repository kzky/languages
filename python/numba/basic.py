# coding: utf-8
import numpy
from numba.decorators import jit
import time

@jit
def pairwise_numba(X):
    M = X.shape[0]
    N = X.shape[1]
    D = numpy.empty((1000, 1000))
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = numpy.sqrt(d)

    return D

def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = numpy.empty((1000, 1000))
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = numpy.sqrt(d)

    return D

if __name__ == '__main__':
    
    t = time.time()
    X = numpy.random.random((1000, 3))
    pairwise_numba(X)
    print "numba:", time.time() - t

    t = time.time()
    X = numpy.random.random((1000, 3))
    pairwise_python(X)
    print "numpy:", time.time() - t
