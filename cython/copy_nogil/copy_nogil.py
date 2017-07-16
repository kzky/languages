import pyximport; pyximport.install()
import numpy as np
from multiprocessing.pool import ThreadPool
import time
from c_copy_nogil import copy_array_nogil
import argparse

def copy(args):
    print(args)
    # Settings
    b, c, h, w = args.batch_size, 3, 224, 224
    n = args.num_threads
    x_list = []
    y_list = []
    pool = ThreadPool(processes=n)

    # Create container
    for _ in range(n):
        x = np.random.rand(b, c, h, w)
        y = np.random.rand(b, c, h, w)
        x_list.append(x)
        y_list.append(y)

    # Parallel copy
    results = []
    st = time.time()
    for i in range(n):
        res = pool.apply_async(copy_array_nogil, (x_list[i], y_list[i]))
        results.append(res)

    # Shutdown pool
    pool.close()
    pool.join()
    
    et = time.time() - st
    print("{}[s]".format(et))

    # Check value
    print(x_list[0][4,1,110,7])
    print(y_list[0][4,1,110,7])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", '--number-threads', type=int, 
                        dest="num_threads", default=16)
    parser.add_argument("-b", '--batch-size', type=int, 
                        dest="batch_size", default=16)
    args = parser.parse_args()

    copy(args)

if __name__ == '__main__':
    main()
    
