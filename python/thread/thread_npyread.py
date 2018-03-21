import numpy as np
import time
import sys
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
import os

def create_data(n_files, shape=(64, 3, 128, 128)):
    for i in range(n_files):
        if os.path.exists("file_{:05d}.npy".format(i)):
            continue
        x = np.random.randn(*shape)
        np.save("file_{:05d}.npy".format(i), x)
    

def read_file(i):
    data = np.load("file_{:05d}.npy".format(i))
    return data

def main():
    # Create files
    n_files = 100
    create_data(n_files)

    # Read with n threads
    n_cpus = 8
    pool = ThreadPool(n_cpus)
    print("NOTE: Do $ sudo sh -c 'echo 3 >/proc/sys/vm/drop_caches' before running.")
    print("num. of cpus is {}".format(n_cpus))
    results = []

    st = time.time()
    for i in range(n_files):
        ret = pool.apply_async(read_file, (i, ))
        results.append(ret)

    for i in range(n_files):
        results[i].get()

    print("Elapsed Time: {} [s]".format(time.time() - st))

if __name__ == '__main__':
    main()
