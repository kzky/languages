class PythonObject(object):

    def __init__(self, a=100):
        self.a = a

cdef class Arithmetic:
    cdef int a, b
    cdef object python_object

    def __cinit__(self, int a, int b=1, *args, **kwargs):
        self.python_object = PythonObject(1000)

    def __init__(self, a, b=1, *args, **kwargs):
        self.a = a
        self.b = b

    cpdef  int add(self, int a, int b):
        return a + b

    cpdef int sub(self, int a, int b):
        return a - b

    cpdef int mul(self, int a, int b):
        return a * b

    cpdef int dev(self, int a, int b):
        return a / b
    
    property python_object:
        def __get__(self):
            return self.python_object

        def __set__(self, python_object):
            self.python_object = python_object

        def __del__(self):
            del self.python_object
    

