#!/usr/bin/env python


class Graph(object):
    
    def __init__(self, ):
        """Directed Acyclic Graph
        Graph represents a set of Vertex and Edge, or G = (V, E).
        """
        self.vertices = set()
        self.edges = set()

class Edge(object):
    """Edge
    """
    
    def __init__(self, name=None):
        """
        """
        self._graph = None

        self.parent_vertices = []
        self.child_vertex = None
        self.name = name

    def __call__(self, *vertex):
        return self.add(*vertex)
        
    def add(self, *vertex):
        """Add vertices and return a new vertex
        """
        new_vertex = Vertex()
        new_vertex.parent_edge = self
        self.child_vertex = new_vertex

        print len(vertex)
        
        if len(vertex) > 1:
            for vertex_ in vertex:
                vertex_.child_edges.append(self)
                self.parent_vertices.append(vertex_)
        else:
            vertex[0].child_edges.append(self)
            self.parent_vertices.append(vertex[0])

        return new_vertex
        
class Vertex(object):
    """Vertex
    """
    
    def __init__(self, name=None):
        """
        """
        self._graph = None

        self.parent_edge = None
        self.child_edges = []
        self.name = name
        
def main():

    v0 = Vertex("input")
    v1 = Edge()(v0)
    v2 = Edge()(v1)
    v3 = Edge()(v2, v1)

    print v3.parent_edge
    
    pass

if __name__ == '__main__':
    main()        

        

        
