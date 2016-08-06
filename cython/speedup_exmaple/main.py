import time
import integrate_py
import pyximport; pyximport.install(pyimport = True)
import integrate_cython
import integrate_cdef_cython
import integrate_cdef_v2_cython

def main():
    a = -100.
    b = 100.
    N = 1000000

    st = time.time()
    print(integrate_py.integrate_f(a, b, N))
    et_python = time.time() - st
    print("Python: {} [s]".format(et_python))

    st = time.time()
    print(integrate_cython.integrate_f(a, b, N))
    et_cython = time.time() - st
    print("Cython: {} [s]".format(et_cython))

    st = time.time()
    print(integrate_cdef_cython.integrate_f(a, b, N))
    et_cdef_cython = time.time() - st
    print("cdef Cython: {} [s]".format(et_cdef_cython))

    st = time.time()
    print(integrate_cdef_v2_cython.integrate_f(a, b, N))
    et_cdef_v2_cython = time.time() - st
    print("cdef v2 Cython: {} [s]".format(et_cdef_v2_cython))

    print("\n### Ratio Comparison### ")
    print("Python: {} [s]".format(et_python/et_python))
    print("Cython: {} [s]".format(et_python/et_cython))
    print("cdef Cython: {} [s]".format(et_python/et_cdef_cython))
    print("cdef v2 Cython: {} [s]".format(et_python/et_cdef_v2_cython))
    

if __name__ == "__main__":
    main()

