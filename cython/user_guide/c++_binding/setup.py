from distutils.core import setup
from distutils.core import Extension
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(
        Extension(
            "rect",                                                             # cython module
            sources=["rect.pyx", "Rectangle.cpp"],  # pyx/c++ codes
            language="c++",                                           # lang-specified
)))
