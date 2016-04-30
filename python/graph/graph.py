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

    def __call__(self, *vertices):
        return self.add(*vertices)
        
    def add(self, *vertices):
        """Add vertices and return a new vertex
        """
        # Create a new vertex,
        # then add an edge as input and add the vertex as output
        new_vertex = Vertex(name="output-{}".format(self.name))
        new_vertex.input_edge = self
        self.output_vertex = new_vertex

        # Add vertices as input and add edges as output
        for vertex in vertices:
            vertex.output_edges.append(self)
            self.input_vertices.append(vertex)

        return new_vertex

    def forward(self, inputs):
        # Concatenation case
        if len(self.input_vertices) > 1:
            if not hasattr(self, "input_cnt"):
                self.input_cnt = len(self.input_vertices)
                self.inputs = []
            self.input_cnt -= 1

            # Return inputs itself
            if self.input_cnt != 0:
                self.inputs.append(inputs)
                return inputs

            # Compute something for each here, just elemwise-sum now
            del self.input_cnt
            self.inputs.append(inputs)
            output = self.infer(self.inputs)
            logger.debug("edge({}).forward".format(self.name))
            return self.output_vertex.forward(output)

        # Compute Inference
        output = self.infer(inputs)

        logger.debug("edge({}).forward".format(self.name))
        return self.output_vertex.forward(output)

    def infer(self, inputs):
        """Infer given inputs

        Parameters
        -----------------
        inputs: ndarray or list of ndarray

        Returns
        -----------
        ndarray
        
        """
        if inputs == []:
            return reduce(lambda x, y: x + y, inputs)

        return inputs
        
    def backward(self, grad):
        logger.debug("edge({}).backword".format(self.name))

        # Compute Gradients
        grads = self.grads(grad, [v.value for v in self.input_vertices])

        grad_ = self.input_vertices[0].backward(grads[0])
        for grad__, input_vertex in zip(grads[1:], self.input_vertices[1:]):
            grad_ += input_vertex.backward(grad__)

        return grad_

    def grads(self, grad, inputs):
        """Grad given input and grad
        Parameters
        -----------------
        grad: ndarray
        inputs: ndarray or list of ndarray

        Returns
        -----------
        list of ndarray
        """
        return [grad * input_ for input_ in inputs]
        
class Vertex(object):
    """Vertex
    """
    
    def __init__(self, name=None):
        self._graph = dag

        self.input_edge = None
        self.output_edges = []
        self.name = name
        self.value = None
        self.grad = None
        
        # Add to graph
        self._graph.vertices.add(self)

    def forward(self, input_):
        logger.debug("vertex({}).forward".format(self.name))

        output_ = input_
        self.value = output_
        if self.output_edges != []:
            return reduce(lambda x, y: x + y,
                          [e.forward(output_) for e in self.output_edges])

        return input_
        
    def backward(self, grad):
        # Concatenation case
        if len(self.output_edges) > 1:
            self.output_cnt = len(self.output_edges) \
                              if not hasattr(self, "output_cnt") else self.output_cnt
            self.output_cnt -= 1

            # Return grad itself
            if self.output_cnt != 0:
                self.grad = grad
                return grad

            # Call backward
            del self.output_cnt
            self.grad += grad
            grad_ = self.grad

            if self.input_edge is not None:
                logger.debug("vertex({}).backward".format(self.name))
                return self.input_edge.backward(grad_)

            # Input vertex case
            return grad

        self.grad = grad

        if self.input_edge is not None:
            logger.debug("vertex({}).backward".format(self.name))
            return self.input_edge.backward(grad)

        return grad

class SquareLoss(Edge):
    
    def __init__(self,  name=None):
        super.__init__(SquareLoss, name=name)
        
    def infer():
        pass
            

        
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
    grad = np.random.rand(10, 5)
    print v5.backward(grad)

if __name__ == '__main__':
    main()

