"""
Result typed input and styped slice numpy is much faster when N=100

Intel(R) Core(TM) i5-2540M CPU @ 2.60GHz

# Ratio Comparison
python: 1.0
cython: 0.868703724676
cython (typed): 1.22492679267
cython (slice typed): 135.697795453
cython (slice typed and more): 6875.01741294
"""

import pyximport; pyximport.install()
import convolution_py
import convolution_pyx
import convolution_pyx_v2
import convolution_pyx_v3
import convolution_pyx_v4
import numpy as np
import time

def main():
    # Image and filter
    N = 1000
    f = np.arange(N*N, dtype=np.int).reshape((N,N))
    g = np.arange(81, dtype=np.int).reshape((9, 9))
    
    # Python
    st = time.time()
    convolution_py.naive_convolve(f, g)
    et_py = time.time() - st
    print("python: {} [s]".format(et_py))

    # Cython
    st = time.time()
    convolution_pyx.naive_convolve(f, g)
    et_pyx = time.time() - st
    print("cpython: {} [s]".format(et_pyx))

    # Cython (typed)
    st = time.time()
    convolution_pyx_v2.naive_convolve(f, g)
    et_pyx_v2 = time.time() - st
    print("cpython (typed): {} [s]".format(et_pyx_v2))

    # Cython (slice typed)
    st = time.time()
    convolution_pyx_v3.naive_convolve(f, g)
    et_pyx_v3 = time.time() - st
    print("cpython (slice typed): {} [s]".format(et_pyx_v3))

    # Cython (slice typed and more)
    st = time.time()
    convolution_pyx_v4.naive_convolve(f, g)
    et_pyx_v4 = time.time() - st
    print("cpython (slice typed and more): {} [s]".format(et_pyx_v4))

    # Ratio Comparison
    print("\n# Ratio Comparison")
    print("python: {}".format(et_py / et_py))
    print("cython: {}".format(et_pyx / et_py))
    print("cython (typed): {}".format(et_py / et_pyx_v2))
    print("cython (slice typed): {}".format(et_py / et_pyx_v3))
    print("cython (slice typed and more): {}".format(et_py / et_pyx_v4))

if __name__ == '__main__':
    main()
