#!/usr/bin/env python

import tensorflow as tf
import datetime

def main():
    # Varialbes
    initial_v1 = tf.truncated_normal((10, 5), mean=0, stddev=0.1)
    #v1 = tf.Variable(initial_v1, name="v1")
    v1 = tf.get_variable("v1", initializer=initial_v1)

    initial_v2 = tf.truncated_normal((5, 3), mean=0, stddev=0.1)
    #v2 = tf.Variable(initial_v2, name="v2")
    v2 = tf.get_variable("v2", initializer=initial_v2)
    tf.add_to_collection("v2", v2)
    
    # Ops
    op_name = "prod"
    prod = tf.matmul(v1, v2, name=op_name)
    init_op = tf.initialize_all_variables()

    # Saver
    saver = tf.train.Saver([v1, v2])
    #saver = tf.train.Saver()

    with tf.Session() as sess:
        # Init variables
        sess.run(init_op)

        # Compute
        result0 = sess.run(prod)
        print result0
        print ""

        # Save variables
        now = datetime.datetime.now()
        save_path = "/tmp/{}-{}.ckpt".format(__file__, now)
        saver.save(sess, save_path, write_meta_graph=True)
        print "Model saved in {}".format(save_path)
        print ""
        
    # Get graph (default graph)
    default_graph = tf.get_default_graph()

    # Save graph
    now = datetime.datetime.now()
    dst_fname = "graph-{}.pbtxt".format(now)
    tf.train.write_graph(default_graph.as_graph_def(),
                         "/tmp", dst_fname,
                         as_text=False)
    print "Graph saved in /tmp/{}".format(dst_fname)
    print ""
        
    # Restore graph to another graph (not default graph)
    graph = tf.Graph()
    with graph.as_default():
        with open("/tmp/{}".format(dst_fname), "rb") as fpin:
            graph_def = graph.as_graph_def()
            graph_def.ParseFromString(fpin.read())
            tf.import_graph_def(graph_def)

    print "Graph restored to another graph (not default graph)"
    print ""
    
    # Restore variables
    with graph.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))

        with tf.Session() as sess:
            saver.restore(sess, save_path)
            print "Variables restored"
            print ""

            print [var.name for var in tf.get_collection(tf.GraphKeys.VARIABLES)]
            print [var.eval() for var in tf.get_collection(tf.GraphKeys.VARIABLES)]

            print tf.get_collection("v2")[0].eval()
            #print tf.get_variable("v2:0") can NOT get
            print ""
            
if __name__ == '__main__':
    main()
