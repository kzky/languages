import pyximport; pyximport.install()
from multi_thread03 import *
import numpy as np
import time
from threading import Thread

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
    q = len(X) // 4
    range_ = range(0, len(X), q)
    threads = []
    for i in range_:
        t = Thread(target=c_array_f, args=(X[i:i+q], ))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    et = time.time()
    print("Elapsed time {} [s]".format(et - st))
    
if __name__ == '__main__':
    main()
