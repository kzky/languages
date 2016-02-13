#!/usr/bin/env python

import tensorflow as tf
import numpy as np


# tf.slice
x = np.random.rand(5, 6, 3)
z_slice = tf.slice(x, [1, 0, 0], [4, 6, 3])


# tf.split
x = np.random.rand(5, 6, 3)
z_split = tf.split(split_dim=1, num_split=3, value=x)


# tf.tile
x = np.random.randint(6, 3)
z_tile = tf.tile(x, multiples=3)


# tf.pad
x = np.random.randint(3, 3)
paddings = np.random.randint(1, 5, (x.ndim, 2))
z_pad = tf.pad(x, paddings=paddings)


# tf.concat
values = [np.random.randint(3, 4, 3)] * 10
z_concat = tf.concat(concat_dim=1, values)

# tf.pack
x = np.random.rand(3, 4)
y = np.random.rand(3, 4)
z = np.random.rand(3, 4)
z_pack = tf.pack([x, y, z])

# tf.unpack
x = np.random.rand(12, 3, 4)
z_unpack = tf.unpack(x)

# tf.reverse_sequence
x = np.random.rand(10, 10, 5, 4)
z_reverse_sequence = tf.reverse_sequence(x,
                                         input=x,
                                         seq_lengths=np.random.randint(0, 10, 10),
                                         seq_dim=1,
                                         batch_dim=0)

# tf.reverse
x = np.random.rand(3, 3, 3, 3)
dims = [False, True, True, False]
z_reverse = tf.reverse(x, dims=dims)

# tf.transpose
x = np.random.rand(3, 3, 3, 3)
perms = [0, 3, 1, 2]
z_transpose = tf.transpose(x, perms=perms)

# tf.gather
x = np.random.rand(10, 4, 3)
indices = [5, 6, 6, 1]
z_gather = tf.gather(x, indices=indices)

# tf.dynamic_partition (similar to segmentation)
x = np.random.rand(10, 4, 3)
partitions = [0, 1, 1, 0, 0, 2, 3, 2, 3, 1]
num_partitions = np.max(partitions) + 1
z_dynamic_partition = tf.dynamic_partition(x, partitions, num_partitions)

# tf.dynamic_stitch
data = [np.random.rand(5, 5, 5)] * 10
indices = np.arange(50)
np.random.shuffle(indices)
indices = np.split(indices, 10) # result is 3d list of 1d-array
z_dynamic_stitch = tf.dynamic_stitch(indices, data)


with tf.Session() as sess:

    print "tf.slice"
    print sess.run(z_slice)

    print "tf.split"
    print sess.run(z_split)

    print "tf.tile"
    print sess.run(z_tile)

    print "tf.pad"
    print sess.run(z_pad)

    print "tf.concat"
    print sess.run(z_concat)

    print "tf.pack"
    print sess.run(z_pack)

    print "tf.unpack"
    print sess.run(z_unpack)

    print "tf.reverse_sequence"
    print sess.run(z_reverse_sequence)

    print "tf.reverse"
    print sess.run(z_reverse)

    print "tf.transpose"
    print sess.run(z_transpose)

    print "tf.gather"
    print sess.run(z_gather)

    print "tf.dynamic_partition"
    print sess.run(z_dynamic_partition)

    print "tf.dynamic_stitch"
    print sess.run(z_dynamic_stitch)



