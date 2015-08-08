#!/usr/bin/env python

"""
To check if diffrence in speed exists.
"""

import cv2
import numpy as np
import glob
import os
from scipy import io
import h5py
import pickle as pkl

def read_with_cv2_to_pkl(n=32):
    base_dirpath = "/home/kzk/datasets/cifar10/train"
    base_save_dirpath = "/home/kzk/datasets/cifar10/train_pkl_{}".format(n)
    if not os.path.exists(base_save_dirpath):
        os.makedirs(base_save_dirpath)

    filepaths = glob.glob("{}/*".format(base_dirpath))
    m = len(filepaths)
    data = []
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:
            print i
        I = cv2.imread(filepath)
        data.append(I)

        if (i + 1) % n == 0:
            filename = os.path.basename(filepath)
            filename = filename.split(".png")[0]
            filename = "{}_{}-{}.pkl".format(filename, i - n, i)
            pkl.dump(I, open("{}/{}".format(base_save_dirpath, filename), "w"))
            del data[:]
        if (i + 1) == m:
            print i
            filename = os.path.basename(filepath)
            filename = filename.split(".png")[0]
            filename = "{}_{}-{}.pkl".format(filename, i - n, i)
            pkl.dump(I, open("{}/{}".format(base_save_dirpath, filename), "w"))

def read_with_cv2_to_mat(n=32):
    base_dirpath = "/home/kzk/datasets/cifar10/train"
    base_save_dirpath = "/home/kzk/datasets/cifar10/train_mat_{}".format(n)
    if not os.path.exists(base_save_dirpath):
        os.makedirs(base_save_dirpath)
    
    filepaths = glob.glob("{}/*".format(base_dirpath))
    m = len(filepaths)
    data = {}
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:
            print i
        I = cv2.imread(filepath)
        data[str(i)] = I

        if (i + 1) % n == 0:
            filename = os.path.basename(filepath)
            filename = filename.split(".png")[0]
            filename = "{}_{}-{}.mat".format(filename, i - n, i)
            io.savemat("{}/{}".format(base_save_dirpath, filename), data)
            data.clear()
        if (i + 1) == m:
            print i
            filename = os.path.basename(filepath)
            filename = filename.split(".png")[0]
            filename = "{}_{}-{}.mat".format(filename, i - n, i)
            io.savemat("{}/{}".format(base_save_dirpath, filename), data)

def read_with_cv2_to_npy(n=32):
    base_dirpath = "/home/kzk/datasets/cifar10/train"
    base_save_dirpath = "/home/kzk/datasets/cifar10/train_npy_{}".format(n)
    if not os.path.exists(base_save_dirpath):
        os.makedirs(base_save_dirpath)

    filepaths = glob.glob("{}/*".format(base_dirpath))
    m = len(filepaths)
    data = []
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:
            print i
        I = cv2.imread(filepath)
        data.append(I)

        if (i + 1) % n == 0:
            filename = os.path.basename(filepath)
            filename = filename.split(".png")[0]
            filename = "{}_{}-{}.npy".format(filename, i - n, i)
            np.save("{}/{}".format(base_save_dirpath, filename), np.asarray(data))
            del data[:]
        if (i + 1) == m:
            print i
            filename = os.path.basename(filepath)
            filename = filename.split(".png")[0]
            filename = "{}_{}-{}.npy".format(filename, i - n, i)
            np.save("{}/{}".format(base_save_dirpath, filename), np.asarray(data))

def read_with_cv2_to_hdf5(n=32):
    base_dirpath = "/home/kzk/datasets/cifar10/train"
    base_save_dirpath = "/home/kzk/datasets/cifar10/train_hdf5_{}".format(n)
    if not os.path.exists(base_save_dirpath):
        os.makedirs(base_save_dirpath)

    filepaths = glob.glob("{}/*".format(base_dirpath))
    m = len(filepaths)
    data = []
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:
            print i
        I = cv2.imread(filepath)
        data.append(I)

        if (i + 1) % n == 0:
            filename = os.path.basename(filepath)
            filename = filename.split(".png")[0]
            filename = "{}_{}-{}.hdf5".format(filename, i - n, i)
            f = h5py.File("{}/{}".format(base_save_dirpath, filename), "w")
            for j, dat in enumerate(data):
                f["data_{}".format(j)] = dat

            f.close()
            del data[:]
            
        if (i + 1) == m:
            print i
            filename = os.path.basename(filepath)
            filename = filename.split(".png")[0]
            filename = "{}_{}-{}.hdf5".format(filename, i - n, i)
            f = h5py.File("{}/{}".format(base_save_dirpath, filename), "w")
            for j, dat in enumerate(data):
                f["data_{}".format(j)] = dat

            f.close()

def main():
    for n in [32, 64, 128]:
        read_with_cv2_to_pkl(n)
        read_with_cv2_to_mat(n)
        read_with_cv2_to_npy(n)
        read_with_cv2_to_hdf5(n)

    pass

if __name__ == '__main__':
    main()
