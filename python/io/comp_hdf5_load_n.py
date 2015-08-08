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

@profile
def load_hdf5_32():
    base_dirpath = "/home/kzk/datasets/cifar10/train_hdf5_32"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        f = h5py.File(filepath, "r")

@profile
def load_hdf5_64():
    base_dirpath = "/home/kzk/datasets/cifar10/train_hdf5_64"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        f = h5py.File(filepath, "r")

@profile
def load_hdf5_128():
    base_dirpath = "/home/kzk/datasets/cifar10/train_hdf5_128"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        f = h5py.File(filepath, "r")


def main():
    
    load_hdf5_32()
    load_hdf5_64()
    load_hdf5_128()

    pass

if __name__ == '__main__':
    main()
