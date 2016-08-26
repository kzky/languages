import pyximport; pyximport.install()
from multi_thread00 import *
import numpy as np
import time

def main():

    X = -1 + 2*np.random.rand(7**2 * 512 * 4096)
    
    print("# Numpy")
    st = time.time()
    Y = array_f(X)
    print(type(Y))
    et = time.time()
    print("Elapsed time {} [s]".format(et - st))

    print("# Cython")
    st = time.time()
    Y = c_array_f(X)
    print(type(np.asarray(Y)))
    et = time.time()
    print("Elapsed time {} [s]".format(et - st))
    
if __name__ == '__main__':
    main()
