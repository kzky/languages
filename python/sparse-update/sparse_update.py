import numpy as np
import time

def sparse_update(size, p=0.07):
    x = np.random.rand(size)
    idx = np.random.choice(np.arange(size), int(size*p), replace=False)
        
    st = time.time()
    x[idx] -= 0.001
    et = time.time() - st
    print("SparseUpdate: {}[s]".format(et))

def main():
    size = int(30 * 1e6)
    p = 0.07
        
    sparse_update(size, p)

if __name__ == '__main__':
    main()

    
    
