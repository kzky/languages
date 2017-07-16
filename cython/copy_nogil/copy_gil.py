import numpy as np
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
import time
import argparse
import threading

def copy_array(x, y):
    x[:] = y
    return None

def copy(args):
    # Settings
    b, c, h, w = args.batch_size, 3, 224, 224
    n = args.num_tasks
    x_list = []
    y_list = []
    pool = ThreadPool(processes=mp.cpu_count())

    # Create container
    for _ in range(n):
        x = np.random.rand(b, c, h, w).astype(np.float32)
        y = np.random.rand(b, c, h, w).astype(np.float32)
        x_list.append(x)
        y_list.append(y)

    # Parallel copy
    results = []
    st = time.time()
    for i in range(n):
        res = pool.apply_async(copy_array, (x_list[i], y_list[i]))
        results.append(res)

    pool.close()
    pool.join()
    
    et = time.time() - st
    print("{}[s]".format(et))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", '--number-tasks', type=int, 
                        dest="num_tasks", default=16)
    parser.add_argument("-b", '--batch-size', type=int, 
                        dest="batch_size", default=16)
    args = parser.parse_args()

    copy(args)

if __name__ == '__main__':
    main()
    
