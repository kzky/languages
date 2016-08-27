import pyximport; pyximport.install()
from arithmetic import *

def main():
    
    ari = Arithmetic(-10)
    a = 10
    b = 3
    print(ari.add(a, b))
    print(ari.sub(a, b))
    print(ari.mul(a, b))
    print(ari.dev(a, b))

    print(ari.python_object.a)
    ari.python_object = PythonObject(-1000)
    print(ari.python_object.a)

if __name__ == '__main__':
    main()
