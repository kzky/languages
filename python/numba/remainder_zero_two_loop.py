#!/usr/bin/env python

import time
import numpy as np
from numba import jit

def remainder_zero(a, d):
    st = time.time()
    for i in a:
        for j in a:
            if i % d == 0:
                pass
    et = time.time()
    print "remainder_zero: elapsed time: {} [s]".format(et - st)

@jit
def remainder_zero_of_list(a, d):
    if type(a) is list:
        a = np.asarray(a)
    st = time.time()
    for i in a:
        for j in a:
            if i % d == 0:
                pass

    et = time.time()
    print "remainder_zero_of_list: elapsed time: {} [s]".format(et - st)

@jit("(i8[:],i4)")
def remainder_zero_of_array(a, d):
    st = time.time()
    for i in a:
        for j in a:
            if i % d == 0:
                pass
    et = time.time()
    print "remainder_zero_of_array: elapsed time: {} [s]".format(et - st)

def main():
    a = np.arange(0, 60000)
    d = 3

    # non jit
    remainder_zero(a, d)

    # list
    remainder_zero_of_list(list(a), d)

    # array
    remainder_zero_of_array(a, d)
    
if __name__ == '__main__':
    main()
    
