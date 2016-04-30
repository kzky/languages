"""Graph module is not thread safe due to state.
"""
import numpy as np

class Graph(object):
    
    def __init__(self, ):
        """Directed Acyclic Graph
        Graph represents a set of Vertex and Edge, or G = (V, E).
        """
        self.vertices = set()
        self.edges = set()

    def find_vertices(self, name=None):
        """Find
        """
        pass
        
    def find_edges(self, name=None):
        """Find
        """
        pass

dag = Graph()

class Edge(object):
    """Edge
    """
    
    def __init__(self, name=None):
        """
        """
        self._graph = dag

        self.input_vertices = []
        self.output_vertex = None
        self.name = name

        # Add to graph
        self._graph.edges.add(self)

    def __call__(self, *vertex):
        return self.add(*vertex)
        
    def add(self, *vertex):
        """Add vertices and return a new vertex
        """
        # Create a new vertex,
        # then add an edge as input and add the vertex as output
        new_vertex = Vertex(name="output-{}".format(self.name))
        new_vertex.input_edge = self
        self.output_vertex = new_vertex

        # Add vertices as input and add edges as output
        for vertex_ in vertex:
            vertex_.output_edges.append(self)
            self.input_vertices.append(vertex_)

        return new_vertex

    def forward(self, inputs):
        # Concatenation case
        if len(self.input_vertices) > 1:
            self.input_cnt = len(self.input_vertices) \
                             if not hasattr(self, "input_cnt") else self.input_cnt
            self.input_cnt -= 1
            self.inputs_past = [] \
                                if not hasattr(self, "inputs_past") else self.inputs_past
            if self.input_cnt == 0:
                self.inputs_past.append(inputs)

                # Compute something here
                outputs = list(self.inputs_past)

                del self.input_cnt
                del self.inputs_past
                
                print "edge({}).forward".format(self.name)
                return self.output_vertex.forward(outputs)
            else:
                self.inputs_past.append(inputs)
                return inputs

        # Compute something here
        outputs = inputs

        print "edge({}).forward".format(self.name)        
        return self.output_vertex.forward(outputs)
        
    def backward(self, grads):
        # Compute something foreach here
        grads_ = np.sum([v.backward(grads) for v in self.input_vertices])

        print "edge({}).forward".format(self.name)
        return grads_
        
class Vertex(object):
    """Vertex
    """
    
    def __init__(self, name=None):
        """
        """
        self._graph = dag

        self.input_edge = None
        self.output_edges = []
        self.name = name
        self.value = None
        self.grads = None
        
        # Add to graph
        self._graph.vertices.add(self)

    def forward(self, inputs):
        outputs_ = inputs
        self.value = outputs_

        print "vertex({}).forward".format(self.name)
        return np.sum([e.forward(outputs_) for e in self.output_edges])
        
    def backward(self, grads):
        # Concatenation case
        if len(self.output_edges) > 1:
            self.output_cnt = len(self.output_edges) \
                              if not hasattr(self, "output_cnt") else self.output_cnt
            self.output_cnt -= 1
            self.grads_past = [] \
                                if not hasattr(self, "grads_past") else self.grads_past
            if self.output_cnt == 0:
                self.grads_past.append(grads)
                grads_ = list(self.grads_past)
                self.grads = grads_
                grads_ = np.sum(grads_)
                
                del self.output_cnt
                del self.grads_past

                print "vertex({}).backward".format(self.name)
                return self.input_edge.backward(grads_)
            else:
                self.grads_past.append(grads)
                return grads

        self.grads = grads

        print "vertex({}).backward".format(self.name)
        return self.input_edge.backward(grads)                
        
def main():

    v0 = Vertex("x")
    v1 = Edge(name="e0")(v0)
    v2 = Edge(name="e1")(v1)
    v3 = Edge(name="e2")(v2, v1)
    v4 = Edge(name="e3")(v3)
    v5 = Edge(name="e4")(v4)

    print "----- Vertices and Edges in Graph -----"
    print len(dag.vertices)
    print len(dag.edges)

    print "----- Forward pass -----"
    print v0.forward()

    print "----- Backward pass -----"
    print v5.backward()

if __name__ == '__main__':
    main()

