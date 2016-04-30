#!/usr/bin/env python


class Graph(object):
    
    def __init__(self, ):
        """Directed Acyclic Graph
        Graph represents a set of Vertex and Edge, or G = (V, E).
        """
        self.vertices = set()
        self.edges = set()

dag = Graph()

class Edge(object):
    """Edge
    """
    
    def __init__(self, name=None):
        """
        """
        self._graph = dag

        self.parent_vertices = []
        self.child_vertex = None
        self.name = name

        # Add to graph
        self._graph.edges.add(self)

    def __call__(self, *vertex):
        return self.add(*vertex)
        
    def add(self, *vertex):
        """Add vertices and return a new vertex
        """
        # Create new vertex, then add edge as parent and add vertex as child
        new_vertex = Vertex()
        new_vertex.parent_edge = self
        self.child_vertex = new_vertex

        # Add vertices as parent and add edges as child
        for vertex_ in vertex:
            vertex_.child_edges.append(self)
            self.parent_vertices.append(vertex_)

        return new_vertex
        
class Vertex(object):
    """Vertex
    """
    
    def __init__(self, name=None):
        """
        """
        self._graph = dag

        self.parent_edge = None
        self.child_edges = []
        self.name = name

        # Add to graph
        self._graph.vertices.add(self)

    def forward():
        pass

    def backward():
        pass
        
def main():

    v0 = Vertex("input")
    v1 = Edge()(v0)
    v2 = Edge()(v1)
    v3 = Edge()(v2, v1)
    v4 = Edge()(v3)
    v5 = Edge()(v4)
    
    print len(dag.vertices), dag.vertices
    print len(dag.edges), dag.edges

if __name__ == '__main__':
    main()        

        

        
