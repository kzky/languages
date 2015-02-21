import numpy as np
import scipy as sp
import logging as logger
import time
from collections import defaultdict

class Kernels(object):
    """
    Kernels Singleton Class.
    """
    instance = None
    
    def print_usage():
        pass
    
    def __new__(cls, *args, **kwargs):
        """
        for making instance as singleton
        """
        if cls.instance is None:
            cls.instance = super(Kernels, cls).__new__(cls, *args, **kwargs)
        return cls.instance

    def sigmoid(self, a = 1, b = 1):
        """
        sigmoid kernel.
        k(x, y) = tanh(a * x^T y + b)
        """
        sigmoid = lambda x, y, a = a, b = b: np.tanh(a * x.dot(y) + b)
        return sigmoid
                        
    def rbf(self, sigma = 1):
        """
        rbf kernel.
        k(x, y) = exp(- sigma * || x - y ||_2^2)
        """
        
        rbf = lambda x, y, sigma=sigma: np.exp(-sigma * sum((x-y)**2))
        return rbf
        
    def polynomial(self, a = 1, b = 0, c = 0):
        """
        polynomial kernel.
        k(x, y) = (a * x^T y + b) ^ c
        """
        polynomial = lambda x, y, a=a, b=b,c=c: (a * x.dot(y) + b) ** c
        return polynomial

    def linear(self, ):
        """
        linear kernel
        k(x, y) = x^T y
        """
        linear = lambda x, y: x.dot(y)
        return linear

    @classmethod
    def main(cls, ):
        kernel = Kernels()
        dim = 10
        x = np.random.normal(0, 1, dim)
        y= np.random.normal(0, 1, dim)

        linear = kernel.linear()
        rbf = kernel.rbf(1)
        polynomial = kernel.polynomial(1, 1, 1)
        sigmoid = kernel.sigmoid(1, 1)
        print "linear:", linear(x, y)
        print "rbf:", rbf(x, y)
        print "polynomial:", polynomial(x, y)
        print "sigmoid:", sigmoid(x, y)
    
if __name__ == '__main__':
    Kernels.main()

