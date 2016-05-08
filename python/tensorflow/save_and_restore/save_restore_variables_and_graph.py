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
    op_name = "prod"
    prod = tf.matmul(v1, v2, name=op_name)
    init_op = tf.initialize_all_variables()

    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Init variables
        sess.run(init_op)

        # Compute
        result0 = sess.run(prod)
        print result0

        # Save variables
        now = datetime.datetime.now()
        save_path = "/tmp/{}-{}.ckpt".format(__file__, now)
        saver.save(sess, save_path)
        print "Model saved in {}".format(save_path)

    # Get graph (default graph)
    default_graph = tf.get_default_graph()

    # Save graph
    now = datetime.datetime.now()
    dst_fname = "graph-{}.pbtxt".format(now)
    tf.train.write_graph(default_graph.as_graph_def(),
                         "/tmp", dst_fname,
                         as_text=False)
    print "Graph saved in /tmp/{}".format(dst_fname)

    # Restore graph to another graph (not default graph)
    graph = tf.Graph()
    with graph.as_default():
        with open("/tmp/{}".format(dst_fname), "rb") as fpin:
            graph_def = graph.as_graph_def()
            graph_def.ParseFromString(fpin.read())
            tf.import_graph_def(graph_def)

    # Restore variables
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_path)

    for var in tf.all_variables():
        graph.add_to_collection(tf.GraphKeys.VARIABLES, var)

    #print [op.name for op in graph.get_operations()]

    with graph.as_default():
        with tf.Session() as sess:
            #tf.initialize_variables(tf.all_variables())
            init_op = graph.get_operation_by_name("import/Variable")
            init_op1 = graph.get_operation_by_name("import/Variable_1")
            sess.run(init_op)
            sess.run(init_op1)
            
            # Get prod op, note 'import' prefix appended
            prod_op = graph.get_operation_by_name("import/{}".format(op_name))
            result1 = sess.run(prod_op)

            print result1
            
if __name__ == '__main__':
    main()
