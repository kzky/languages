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
def load_mat_32():
    base_dirpath = "/home/kzk/datasets/cifar10/train_mat_32"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        io.loadmat(filepath)

@profile
def load_mat_64():
    base_dirpath = "/home/kzk/datasets/cifar10/train_mat_64"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        io.loadmat(filepath)

@profile
def load_mat_128():
    base_dirpath = "/home/kzk/datasets/cifar10/train_mat_128"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        io.loadmat(filepath)

def main():
    load_mat_32()
    load_mat_64()
    load_mat_128()

    pass

if __name__ == '__main__':
    main()
