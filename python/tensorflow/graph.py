#!/usr/bin/env python

import tensorflow as tf
import threading
import numpy as np

class GraphWorker(threading.Thread):
    """
    """
    
    def __init__(self, index):
        """
        """
        super(GraphWorker, self).__init__()
        self.index = index
        
        pass

    def run(self):

        g = tf.get_default_graph()
        #print "Graph in main thread", g

        g_local = tf.Graph()
        #print "Graph in this thread", g_local

def main():
    # Default graph
    """
    Confirm that a default Graph is always registered, and accessible by calling tf.get_default_graph(). To add an operation to the default graph, simply call one of the functions that defines a new Operation:
    """
    c = tf.constant(4.0)
    assert c.graph is tf.get_default_graph()

    # Create another graph in this thread (main thread)
    """
    Confirm that tf.Graph.as_default() method should be used if you want to create multiple graphs in the same process.
    """
    with tf.Graph().as_default() as g:
        c = tf.constant(30.0)
        assert c.graph is g
        d = tf.constant(40.0)
        assert d.graph is g
        e = tf.Variable(np.random.rand(10, 10), name="e")
        
    assert tf.get_default_graph() != g

    # Graph in multi thread
    """
    The default graph is a property of the current thread. If you create a new thread, and wish to use the default graph in that thread, you must explicitly add a with g.as_default(): in that thread's function.
    """
    n = 4
    graph_workers = []
    for i in xrange(n):
        worker = GraphWorker(i)
        worker.start()
        graph_workers.append(worker)

    for i in xrange(n):
        graph_workers[i].join()

    # Write graph as protbuf to disk
    print g.get_operations()
    print len(g.get_operations())
    tf.train.write_graph(g.as_graph_def(), "./graph_dir_text", "./graph.pbtxt")
    tf.train.write_graph(g.as_graph_def(), "./graph_dir", "./graph.pb", as_text=False)

    # Read graph from disk and to Graph
    print tf.get_default_graph()
    with open("./graph_dir/graph.pb", "rb") as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def)
    print tf.get_default_graph()

    print "----- ops in g -----"
    for op in g.get_operations():
        print op.name

    print "----- ops in default graph after import_graph_def -----"
    for op in tf.get_default_graph().get_operations():
        print op.name
    
    pass


if __name__ == '__main__':
    main()
