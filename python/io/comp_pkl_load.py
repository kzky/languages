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
def load_pkl():
    base_dirpath = "/home/kzk/datasets/cifar10/train_pkl"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:
            print i
        pkl.load(open(filepath))

def main():
    load_pkl()
    pass

if __name__ == '__main__':
    main()
