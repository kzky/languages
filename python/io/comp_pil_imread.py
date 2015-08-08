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

@profile
def read_with_pil():
    """
    """

    base_dirpath = "/home/kzk/datasets/cifar10/train"
    filepaths = glob.glob("{}/*".format(base_dirpath))
    for i, filepath in enumerate(filepaths):
        if i % 1000 == 0:
            print i
        I = np.asarray(Image.open(filepath))
        
def main():
    read_with_pil()
    pass

if __name__ == '__main__':
    main()
    
