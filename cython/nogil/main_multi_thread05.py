import pyximport; pyximport.install(language_level=2)
from multi_thread05 import *
import numpy as np
import time

def main():

    X = -1 + 2*np.random.rand(7**2 * 512 * 4096)
    n = 10
    
    print("# Numpy")
    t = []
    for _ in range(n):
        st = time.time()
        Y = array_f(X)
        et = time.time()
        t.append(et - st)
    print("Elapsed time (ave) {} [s]".format(np.mean(t)))
    print("Elapsed time (std) {} [s]".format(np.std(t)))

    print("# Cython")
    t = []
    for _ in range(n):
        st = time.time()
        q = len(X) // 4
        range_ = range(0, len(X), q)
        threads = []
        for i in range_:
            tt = TaskThread(X[i:i+q])
            tt.start()
            threads.append(tt)
     
        Y = []
        for tt in threads:
            tt.join()
            Y.append(tt.y)
        #Y = np.concatenate(Y)
        et = time.time()
        t.append(et - st)

    print("Elapsed time (ave) {} [s]".format(np.mean(t)))
    print("Elapsed time (std) {} [s]".format(np.std(t)))
    
if __name__ == '__main__':
    main()
