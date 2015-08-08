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
import pickle as pkl

@profile
def load_pkl_32():
    base_dirpath = "/home/kzk/datasets/cifar10/train_pkl_32"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        pkl.load(open(filepath))

@profile
def load_pkl_64():
    base_dirpath = "/home/kzk/datasets/cifar10/train_pkl_64"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        pkl.load(open(filepath))

@profile
def load_pkl_128():
    base_dirpath = "/home/kzk/datasets/cifar10/train_pkl_128"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        pkl.load(open(filepath))

def main():
    load_pkl_32()
    load_pkl_64()
    load_pkl_128()

if __name__ == '__main__':
    main()
