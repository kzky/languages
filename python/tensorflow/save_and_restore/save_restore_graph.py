#!/usr/bin/env python

import tensorflow as tf
import datetime

def main():
    # Varialbes
    initial_v1 = tf.truncated_normal((10, 5), mean=0, stddev=0.1)
    v1 = tf.Variable(initial_v1)

    initial_v2 = tf.truncated_normal((5, 3), mean=0, stddev=0.1)
    v2 = tf.Variable(initial_v2)

    # Ops
    prod = tf.matmul(v1, v2)
    init_ops = tf.initialize_all_variables()

    # Get graph (default graph)
    default_graph = tf.get_default_graph()
    print "----- Graph operations of default graph -----"
    print default_graph.get_operations()
    print ""

    # Save graph
    now = datetime.datetime.now()
    dst_fname = "graph.pbtxt".format(now)
    tf.train.write_graph(default_graph.as_graph_def(),
                         "/tmp", dst_fname,
                         as_text=False)
    
    # Restore graph to another graph (not default graph)
    with open("/tmp/{}".format(dst_fname), "rb") as fpin:
        graph = tf.Graph()
        with graph.as_default():
            graph_def = graph.as_graph_def()
            graph_def.ParseFromString(fpin.read())
            tf.import_graph_def(graph_def)

    print "----- Graph operations of another graph -----"
    print graph.get_operations()
    print ""

    default_graph_op_names = []
    for op in default_graph.get_operations():
        default_graph_op_names.append(op.name)
    default_graph_op_names.sort()

    graph_op_names = []
    for op in graph.get_operations():
        graph_op_names.append(op.name.split("import/")[-1])
    graph_op_names.sort()

    print default_graph_op_names == graph_op_names
        
if __name__ == '__main__':
    main()
