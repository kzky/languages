#!/usr/bin/env python

"""
To check if diffrence in speed exists.
"""

from PIL import Image
from scipy.misc import imread
import cv2
import numpy as np
import glob
import time
import os
from scipy import io
import h5py
import pickle as pkl

def read_with_cv2_to_pkl():
    base_dirpath = "/home/kzk/datasets/cifar10/train"
    base_save_dirpath = "/home/kzk/datasets/cifar10/train_pkl"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:
            print i
        I = cv2.imread(filepath)
        filename = os.path.basename(filepath)
        filename = filename.split(".png")[0]
        filename = "{}.pkl".format(filename)
        pkl.dump(I, open("{}/{}".format(base_save_dirpath, filename), "w"))

def read_with_cv2_to_mat():
    base_dirpath = "/home/kzk/datasets/cifar10/train"
    base_save_dirpath = "/home/kzk/datasets/cifar10/train_mat"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:
            print i
        I = cv2.imread(filepath)
        dict_I = dict(data=I)
        filename = os.path.basename(filepath)
        filename = filename.split(".png")[0]
        filename = "{}.mat".format(filename)
        io.savemat("{}/{}".format(base_save_dirpath, filename), dict_I)

def read_with_cv2_to_npy():
    base_dirpath = "/home/kzk/datasets/cifar10/train"
    base_save_dirpath = "/home/kzk/datasets/cifar10/train_npy"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:
            print i
        I = cv2.imread(filepath)
        filename = os.path.basename(filepath)
        filename = filename.split(".png")[0]
        filename = "{}.npy".format(filename)
        np.save("{}/{}".format(base_save_dirpath, filename), I)

def read_with_cv2_to_hdf5():
    base_dirpath = "/home/kzk/datasets/cifar10/train"
    base_save_dirpath = "/home/kzk/datasets/cifar10/train_hdf5"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:
            print i
        I = cv2.imread(filepath)
        filename = os.path.basename(filepath)
        filename = filename.split(".png")[0]
        filename = "{}.hdf5".format(filename)
        f = h5py.File("{}/{}".format(base_save_dirpath, filename), "w")
        #f.create_dataset("mydataset", I)
        f["mydataset"] = I
        f.close()

def main():
    read_with_cv2_to_pkl()
    read_with_cv2_to_mat()
    read_with_cv2_to_npy()
    read_with_cv2_to_hdf5()

    pass

if __name__ == '__main__':
    main()
