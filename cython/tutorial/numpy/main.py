import pyximport; pyximport.install()
import convolution_py
import convolution_pyx
import convolution_pyx_v2
import convolution_pyx_v3
import convolution_pyx_v4
import numpy as np
import time

def main():
    # Python
    st = time.time()
    convolution_py.naive_convolve(np.array([[1, 1, 1]], dtype=np.int),
                               np.array([[1],[2],[1]], dtype=np.int))
    et_py = time.time() - st
    print("python: {} [s]".format(et_py))

    # Cython
    st = time.time()
    convolution_pyx.naive_convolve(np.array([[1, 1, 1]], dtype=np.int),
                            np.array([[1],[2],[1]], dtype=np.int))
    et_pyx = time.time() - st
    print("cpython: {} [s]".format(et_pyx))

    # Cython (typed)
    st = time.time()
    convolution_pyx_v2.naive_convolve(np.array([[1, 1, 1]], dtype=np.int),
                            np.array([[1],[2],[1]], dtype=np.int))
    et_pyx_v2 = time.time() - st
    print("cpython (typed): {} [s]".format(et_pyx_v2))

    # Cython (slice typed)
    st = time.time()
    convolution_pyx_v3.naive_convolve(np.array([[1, 1, 1]], dtype=np.int),
                            np.array([[1],[2],[1]], dtype=np.int))
    et_pyx_v3 = time.time() - st
    print("cpython (slice typed): {} [s]".format(et_pyx_v3))

    # Cython (slice typed and more)
    st = time.time()
    convolution_pyx_v4.naive_convolve(np.array([[1, 1, 1]], dtype=np.int),
                            np.array([[1],[2],[1]], dtype=np.int))
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
