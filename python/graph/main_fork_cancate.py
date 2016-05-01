#!/usr/bin/env python

import numpy as np
from graph import Edge, Vertex, SquareLoss, dag

def main():
    # Fork and concate
    v_in = Vertex("x")  # single source
    v1 = Edge(name="e0")(v_in)
    v2 = Edge(name="e1")(v1)
    v3 = Edge(name="e2")(v2, v1)  # fork and concate
    v4 = Edge(name="e3")(v3)
    v5 = Edge(name="e4")(v4)

    print "----- Vertices and Edges in Graph -----"
    print len(dag.vertices)
    print len(dag.edges)

    print "----- Forward pass (Inference) -----"
    inputs = np.random.rand(4, 3)
    v_in.forward(inputs)
    print v1.value
    print v5.value

    print "----- Backward pass (from the middle) -----"
    grad = np.random.rand(4, 3)
    v3.backward(grad)
    print v1.grad
     
if __name__ == '__main__':
    main()
