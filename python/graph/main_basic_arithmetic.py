#!/usr/bin/env python

import numpy as np
from graph import Vertex


def main():

    # Additon
    print "# Additon"
    v = np.asarray(np.arange(1, 10)).reshape(3, 3).astype(np.float32)
    v0 = Vertex(name="", value=v)
    v1 = Vertex(name="", value=v)
    v2 = v0 + v1
    v0.forward(), v1.forward()
    print v2.value
    v2.backward()
    print v0.grad
    print v1.grad
    print ""
    
    # Subtraction
    print "# Subtraction"
    v = np.asarray(np.arange(1, 10)).reshape(3, 3).astype(np.float32)
    v0 = Vertex(name="", value=v)
    v1 = Vertex(name="", value=v)
    v2 = v0 - v1
    v0.forward(), v1.forward()
    print v2.value
    v2.backward()
    print v0.grad
    print v1.grad
    print ""
    
    # Multiplication
    print "# Multiplication"
    v = np.asarray(np.arange(1, 10)).reshape(3, 3).astype(np.float32)
    v0 = Vertex(name="", value=v)
    v1 = Vertex(name="", value=v)
    v2 = v0 * v1
    v0.forward(), v1.forward()
    print v2.value
    v2.backward()
    print v0.grad
    print v1.grad
    print ""

    # Division
    print "# Division"
    v = np.asarray(np.arange(1, 10)).reshape(3, 3).astype(np.float32)
    v0 = Vertex(name="", value=v)
    v1 = Vertex(name="", value=v)
    v2 = v0 / v1
    v0.forward(), v1.forward()
    print v2.value
    v2.backward()
    print v0.grad
    print v1.grad
    print ""

if __name__ == '__main__':
    main()
