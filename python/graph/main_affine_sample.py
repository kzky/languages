#!/usr/bin/env python

import numpy as np
from graph import Edge, Vertex, Affine, SquareLoss, cg

def main():
    # Two sources and single distination
    v_in = Vertex("x")  # one source
    v1_w = Vertex("w1", value=np.random.rand(10, 5))
    v1 = Affine(name="e1")(v1_w, v_in)
    v2_w = Vertex("w2", value=np.random.rand(5, 1))
    v2 = Affine(name="e2")(v2_w, v1)
    y = Vertex("y")  # second source
    v_out = SquareLoss(name="square-loss")(v2, y)  # single distination

    print "----- Vertices and Edges in Graph -----"
    print len(cg.vertices)
    print len(cg.edges)

    print "----- Forward pass (Inference) -----"
    inputs = np.random.rand(100, 10)
    v_in.forward(inputs)
    v1_w.forward()
    v2_w.forward()
    labels = np.random.rand(100, 1)
    y.forward(labels)
    print v2.value

    print "----- Compute Loss -----"
    v_out.backward(1)
    v2_w.grad

if __name__ == '__main__':
    main()
