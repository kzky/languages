"""Graph module is not thread safe due to state.
"""
import numpy as np
import logging

FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
logging.basicConfig(
    format=FORMAT,
    level=logging.DEBUG)
logger = logging.getLogger("CG")

class ComputationalGraph(object):
    #TODO: add forward/backword function in graph
    
    def __init__(self, ):
        """Directed Acyclic Computational Graph
        DAG represents a set of Vertex and Edge, or G = (V, E).
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

cg = ComputationalGraph()
        
class Vertex(object):
    """Vertex
    """
    
    def __init__(self, name=None, value=None):
        self._graph = cg

        self.in_edge = None
        self.out_edges = []
        self.name = name
        self.value = value
        self.grad = None
        
        # Add to graph
        self._graph.vertices.add(self)

    def forward(self, in_=None):
        logger.debug("vertex({}).forward".format(self.name))

        if in_ is None:
            in_ = self.value
        out_ = in_
        self.value = out_
        if self.out_edges != []:
            out__ = None
            for e in self.out_edges:
                e.forward(out_)
            return out__

        return out_
        
    def backward(self, grad=None):
        if grad is None:
            grad = 1
            
        # Concatenation case
        if len(self.out_edges) > 1:
            self.out_cnt = len(self.out_edges) \
                           if not hasattr(self, "out_cnt") else self.out_cnt
            self.out_cnt -= 1

            # Return grad itself
            if self.out_cnt != 0:
                self.grad = grad
                return

            # Call backward
            del self.out_cnt
            self.grad += grad

            if self.in_edge is not None:
                logger.debug("vertex({}).backward".format(self.name))
                self.in_edge.backward(self.grad)
                return

            # Input vertex case
            return

        # Sequence case
        self.grad = grad
        if self.in_edge is not None:
            logger.debug("vertex({}).backward".format(self.name))
            self.in_edge.backward(grad)
            return

        return

    def __add__(self, other):
        v = Add(name="add-{}-{}".format(self.name, other.name))(self, other)
        return v
        
    def __iadd__(self, other):
        return NotImplementedError("iterat")

    def __sub__(self, other):
        v = Sub(name="sub-{}-{}".format(self.name, other.name))(self, other)
        return v
        
    def __isub__(self, other):
        return NotImplementedError("iterat")
        
    def __mul__(self, other):
        v = Mul(name="mul-{}-{}".format(self.name, other.name))(self, other)
        return v
        
    def __imul__(self, other):
        return NotImplementedError("iterat")

    def __div__(self, other):
        v = Div(name="div-{}-{}".format(self.name, other.name))(self, other)
        return v

    def __idiv__(self, other):
        return NotImplementedError("iterat")

class Edge(object):
    """Edge
    """
    
    def __init__(self, name=None):
        self._graph = cg

        self.in_vertices = []
        self.out_vertex = None
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
        new_vertex.in_edge = self
        self.out_vertex = new_vertex

        # Add vertices as input and add edges as output
        for vertex in vertices:
            vertex.out_edges.append(self)
            self.in_vertices.append(vertex)

        return new_vertex

    def forward(self, input_):
        # Concatenation case
        if len(self.in_vertices) > 1:
            if not hasattr(self, "in_cnt"):
                self.in_cnt = len(self.in_vertices)
                self.inputs = []
            self.in_cnt -= 1

            # Return inputs itself
            if self.in_cnt != 0:
                self.inputs.append(input_)
                return

            # Compute something for each here, just elemwise-sum now
            del self.in_cnt
            self.inputs.append(input_)
            output = self.infer(self.inputs)
            logger.debug("edge({}).forward".format(self.name))
            self.out_vertex.forward(output)
            return

        # Sequence case
        # Compute Inference
        output = self.infer(input_)

        logger.debug("edge({}).forward".format(self.name))
        self.out_vertex.forward(output)
        return

    def infer(self, inputs):
        """Infer given inputs

        Parameters
        -----------------
        inputs: ndarray or list of ndarray

        Returns
        ------------
        ndarray
        
        """
        # Default infer is addition
        if type(inputs) == list:
            return reduce(lambda x, y: x + y, inputs)

        return inputs
        
    def backward(self, grad):
        logger.debug("edge({}).backword".format(self.name))

        # Compute Gradients
        grads = self.grads(grad, [v.value for v in self.in_vertices])

        for grad__, in_vertex in zip(grads, self.in_vertices):
            in_vertex.backward(grad__)

    def grads(self, grad, inputs):
        """Grad given input and grad
        Parameters
        -----------------
        grad: ndarray
        inputs: ndarray or list of ndarray

        Returns
        ------------
        list of ndarray
        """
        return [grad * in_ for in_ in inputs]

class Add(Edge):
    def __init__(self, name=None):
        super(Add, self).__init__(name=name)

    def infer(self, inputs):

        if len(inputs) != 2:
            raise ValueError("Input have to be 2")

        return inputs[0] + inputs[1]

    def grads(self, grad, inputs):
        grads = []
        grad0 = np.ones_like(inputs[0])
        grad1 = np.ones_like(inputs[1])
        grads.append(grad0)
        grads.append(grad1)

        return grads

class Sub(Edge):
    def __init__(self, name=None):
        super(Sub, self).__init__(name=name)

    def infer(self, inputs):

        if len(inputs) != 2:
            raise ValueError("Input have to be 2")

        return inputs[0] - inputs[1]

    def grads(self, grad, inputs):
        grads = []
        grad0 = np.ones_like(inputs[0])
        grad1 = - np.ones_like(inputs[1])
        grads.append(grad0)
        grads.append(grad1)

        return grads

class Mul(Edge):
    def __init__(self, name=None):
        super(Mul, self).__init__(name=name)

    def infer(self, inputs):

        if len(inputs) != 2:
            raise ValueError("Input have to be 2")

        return inputs[0] * inputs[1]

    def grads(self, grad, inputs):
        grads = []
        grad0 = inputs[1]
        grad1 = inputs[0]
        grads.append(grad0)
        grads.append(grad1)

        return grads

class Div(Edge):
    def __init__(self, name=None):
        super(Div, self).__init__(name=name)

    def infer(self, inputs):

        if len(inputs) != 2:
            raise ValueError("Input have to be 2")

        return inputs[0] / inputs[1]

    def grads(self, grad, inputs):
        grads = []
        grad0 = np.ones_like(inputs[0]) / inputs[1]
        grad1 = - inputs[0] / (inputs[1] ** 2)
        grads.append(grad0)
        grads.append(grad1)

        return grads

class Affine(Edge):
    def __init__(self, name=None):
        super(Affine, self).__init__(name=name)

    def infer(self, inputs):

        if len(inputs) != 2:
            msg = "Input have to be 2"
            raise ValueError(msg)

        # Get shape
        shape0 = inputs[0].shape
        shape1 = inputs[1].shape

        # Check param shape
        if len(shape0) != 2 and len(shape1) != 2:
            msg = "Tensor dimension of Affine parameter have to be 2"
            raise ValueError(msg)

        input0 = inputs[0]
        if len(shape0) > 2:
            input0 = input0.reshape((shape0, np.prod(shape0[1:])))

        input1 = inputs[1]
        if len(shape1) > 2:
            input1 = input1.reshape((shape1, np.prod(shape1[1:])))
            
        if input0.shape[1] == input1.shape[0]:
            output = input0.dot(input1)
        else:
            output = input1.dot(input0)

        return output

    def grads(self, grad, inputs):
        grads = []
        
        
class SquareLoss(Edge):
    
    def __init__(self, name=None):
        super(SquareLoss, self).__init__(name=name)

    def infer(self, inputs):

        if len(inputs) != 2:
            raise ValueError("Input have to be 2")

        return (inputs[0] - inputs[1]) ** 2

    def grads(self, grad, inputs):
        grads = []
        grad0 = 2 * (inputs[0] - inputs[1])
        grad1 = 2 * (inputs[1] - inputs[0])
        grads.append(grad0)
        grads.append(grad1)

        return grads
