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
def load_hdf5():
    base_dirpath = "/home/kzk/datasets/cifar10/train_hdf5"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:
            print i
            
        f = h5py.File(filepath, "r")
        

def main():
    load_hdf5()

    pass

if __name__ == '__main__':
    main()
