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

@profile
def load_npy_32():
    base_dirpath = "/home/kzk/datasets/cifar10/train_npy_32"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        np.load(filepath)

@profile
def load_npy_64():
    base_dirpath = "/home/kzk/datasets/cifar10/train_npy_64"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        np.load(filepath)

@profile
def load_npy_128():
    base_dirpath = "/home/kzk/datasets/cifar10/train_npy_128"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        np.load(filepath)

def main():
    load_npy_32()
    load_npy_64()
    load_npy_128()
    pass

if __name__ == '__main__':
    main()
