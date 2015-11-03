# -*- coding: utf-8 -*-

"""
1. 1processでnumpyの計算を行う
"""

import numpy as np
import time


def main():

    n = 10000
    m = 100
    X = np.random.rand(n, m)

    st = time.time()
    for i in xrange(n):
        for j in xrange(n):
            X[i, :].dot(X[j, :])
    et = time.time()

    print "Elapsed time {} [s]".format(et - st)

if __name__ == '__main__':
    main()
