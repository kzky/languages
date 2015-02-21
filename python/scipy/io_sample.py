#!/usr/bin/python

## input/output sample
import scipy as sp

fname = "/home/kzk/tmp/iosample.npy"
A = sp.array([[ 1,  2,  3,  4],
       [ 4,  5,  6,  7],
       [ 7,  8,  9, 10],
       [10, 11, 12, 13]])

print A.shape

sp.save(file = fname, arr = A)
B = sp.load(file = fname)
print "B = \n", B






