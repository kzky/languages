"""Graph module is not thread safe due to state.
"""
import numpy as np
import logging

FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
logging.basicConfig(
    format=FORMAT,
    level=logging.DEBUG)
logger = logging.getLogger("DAG")

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

            # Return inputs itself
            if self.input_cnt != 0:
                self.inputs = [inputs]
                return inputs

            # Compute something for each here, just elewise-sum now
            del self.input_cnt
            self.inputs.append(inputs)
            outputs = reduce(lambda x, y: x + y, self.inputs)
            logger.debug("edge({}).forward".format(self.name))

            return self.output_vertex.forward(outputs)

        # Compute something here
        outputs = inputs

        logger.debug("edge({}).forward".format(self.name))
        return self.output_vertex.forward(outputs)
        
    def backward(self, grads):
        # Compute something for each here, just elewise-sum now
        grads_ = reduce(lambda x, y: x + y,
                        [v.backward(grads) for v in self.input_vertices])

        logger.debug("edge({}).forward".format(self.name))
        
        return grads_
        
class Vertex(object):
    """Vertex
    """
    
    def __init__(self, name=None):
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

        logger.debug("vertex({}).forward".format(self.name))
        if self.output_edges != []:
            return reduce(lambda x, y: x + y,
                          [e.forward(outputs_) for e in self.output_edges])

        return inputs
        
    def backward(self, grads):
        # Concatenation case
        if len(self.output_edges) > 1:
            self.output_cnt = len(self.output_edges) \
                              if not hasattr(self, "output_cnt") else self.output_cnt
            self.output_cnt -= 1

            # Return grads itself
            if self.output_cnt != 0:
                self.grads = grads
                return grads

            # Call backward, 
            del self.output_cnt
            self.grads += grads
            grads_ = self.grads

            logger.debug("vertex({}).backward".format(self.name))

            if self.input_edge is not None:
                return self.input_edge.backward(grads_)

            # Input vertex case
            return grads

        self.grads = grads

        logger.debug("vertex({}).backward".format(self.name))
        if self.input_edge is not None:
            return self.input_edge.backward(grads)

        return grads
        
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
    inputs = np.random.rand(10, 5)
    print v0.forward(inputs)

    print "----- Backward pass -----"
    grads = np.random.rand(10, 5)
    print v5.backward(grads)

if __name__ == '__main__':
    main()

