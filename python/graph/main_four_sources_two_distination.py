#!/usr/bin/env python

import numpy as np
from graph import Edge, Vertex, SquareLoss, cg

def main():
    # Two sources and one distination
    v1_in = Vertex("x1")  # one source
    v1_1 = Edge(name="e1_0")(v1_in)
    v1_2 = Edge(name="e1_1")(v1_1)
    v1_3 = Edge(name="e1_2")(v1_2)
    v1_4 = Edge(name="e1_3")(v1_3)
    v1_5 = Edge(name="e1_4")(v1_4)
    y1 = Vertex("y1")  # second source
    y1.value = np.random.rand(4, 3)
    v1_out = SquareLoss(name="square-loss")(v1_5, y1)  # one distination

    # Two sources and one distination
    v2_in = Vertex("x2")  # one source
    v2_1 = Edge(name="e2_0")(v2_in)
    v2_2 = Edge(name="e2_1")(v2_1)
    v2_3 = Edge(name="e2_2")(v2_2)
    v2_4 = Edge(name="e2_3")(v2_3)
    v2_5 = Edge(name="e2_4")(v2_4)
    y2 = Vertex("y2")  # second source
    y2.value = np.random.rand(4, 3)
    v2_out = SquareLoss(name="square-loss")(v2_5, y2)  # one distination

    # Objective function
    v_obj = v1_out + v2_out

    print "----- Vertices and Edges in Graph -----"
    print len(cg.vertices)
    print len(cg.edges)

    print "----- Forward pass (Inference) -----"
    inputs1 = np.random.rand(4, 3)
    inputs2 = np.random.rand(4, 3)
    v1_in.forward(inputs1), v2_in.forward(inputs2)

    labels1 = np.random.rand(4, 3)
    labels2 = np.random.rand(4, 3)
    y1.forward(labels1), y2.forward(labels2)

    print "----- Compute Loss -----"
    v_obj.backward()
    print v1_1.grad
    

if __name__ == '__main__':
    main()
