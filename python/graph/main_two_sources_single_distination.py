#!/usr/bin/env python

import numpy as np
from graph import Edge, Vertex, SquareLoss, cg

def main():
    # Two sources and single distination
    v_in = Vertex("x")  # one source
    v1 = Edge(name="e0")(v_in)
    v2 = Edge(name="e1")(v1)
    v3 = Edge(name="e2")(v2, v1)  # fork and concate
    v4 = Edge(name="e3")(v3)
    v5 = Edge(name="e4")(v4)
    y = Vertex("y")  # second source
    y.value = np.random.rand(4, 3)
    v_out = SquareLoss(name="square-loss")(v5, y)  # single distination

    print "----- Vertices and Edges in Graph -----"
    print len(cg.vertices)
    print len(cg.edges)

    print "----- Forward pass (Inference) -----"
    inputs = np.random.rand(4, 3)
    v_in.forward(inputs)
    labels = np.random.rand(4, 3)
    y.forward(labels)
    print v1.value
    print v5.value

    print "----- Compute Loss -----"
    v_out.backward(1)
    print v1.grad
    

if __name__ == '__main__':
    main()
