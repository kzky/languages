#!/usr/bin/env python

import glob
import time
import pickle

# http://docs.python.jp/2/library/queue.html#module-Queue

# read jpeg to ndarray
# read pickle to ndarray
def read_pkl_2_ndarray(filepath):
    return pickle.load(open(filepath, "r"))

def main():
    base_dirpath = "/home/kzk/datasets/cifar10/train_pkl"
    filepaths = glob.glob("{}/*".format(base_dirpath))

    st = time.time()
    for filepath in filepaths:
        read_pkl_2_ndarray(filepath)
        pass

    et = time.time()
    print "total execution time with threading: ", (et - st), "[s]"
    pass

if __name__ == '__main__':
    main()

    # total execution time with threading:  452.878698111 [s]
