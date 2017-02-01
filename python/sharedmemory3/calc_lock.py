import multiprocessing as mp
from multiprocessing import sharedctypes
import time
from numpy import ctypeslib
import os
import numpy as np

def main():
    size = int(10*1e6)
    array = sharedctypes.Array("f", size)
    max_epoch = 150
    batch_size = 128
    n_samples = int(10 * 10000)
    iter_per_epoch = n_samples / batch_size
    n_iters = max_epoch * iter_per_epoch
        
    st = time.time()
    for i in range(n_iters):
        with array.get_lock():
            pass
    et = time.time() - st
    print("ElapsedTime:{} [s]".format(et))
        
if __name__ == '__main__':
    main()
