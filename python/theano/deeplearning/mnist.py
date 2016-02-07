#!/usr/bin/env python

"""
Download http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz first.

Format is as follows, 

[(x, y)_i]

x: 784-d vector, 1-order tensor
y: vector

"""

import gzip
import cPickle

input_filepath = "/home/kzk/downloads/mnist.pkl.gz"

with gzip.open(input_filepath, "rb") as fpin:
    train_set, valid_set, test_set = cPickle.load(fpin)
    pass
