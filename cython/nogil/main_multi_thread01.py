import pyximport; pyximport.install()
from multi_thread01 import *
import numpy as np
import time

def main():

    X = -1 + 2*np.random.rand(7**2 * 512 * 4096)
    n = 10
        
    print("# Numpy")
    t = []
    for i in range(10):
        st = time.time()
        Y = array_f(X)
        et = time.time()
        t.append(et - st)
    print("Elapsed time (ave) {} [s]".format(np.mean(t)))
    print("Elapsed time (std) {} [s]".format(np.std(t)))

    print("# Cython")
    t = []
    for i in range(10):
        st = time.time()
        Y = c_array_f(X)
        et = time.time()
        t.append(et - st)
    print("Elapsed time (ave) {} [s]".format(np.mean(t)))
    print("Elapsed time (std) {} [s]".format(np.std(t)))
    
if __name__ == '__main__':
    main()
