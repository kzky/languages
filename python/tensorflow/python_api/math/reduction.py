import tensorflow as tf
import numpy as np

# tf.reduce_sum
x = np.random.rand(10, 5, 4)
z_reduce_sum = tf.reduce_sum(x, reduction_indices=[0, 2])

# tf.reduce_prod
x = np.random.rand(10, 5, 4)
z_reduce_prod = tf.reduce_prod(x, reduction_indices=[0, 2])

# tf.reduce_min
x = np.random.rand(10, 5, 4)
z_reduce_min = tf.reduce_min(x, reduction_indices=[0, 2])

# tf.reduce_max
x = np.random.rand(10, 5, 4)
z_reduce_max = tf.reduce_max(x, reduction_indices=[0, 2])

# tf.reduce_mean
x = np.random.rand(10, 5, 4)
z_reduce_mean = tf.reduce_mean(x, reduction_indices=[0, 2])

# tf.reduce_all
x = np.random.randint(0, 2, 10 * 5 * 4)
z = np.empty(10 * 5 * 4, dtype=np.bool)
for i, x_ in enumerate(x):
    if x_ > 0:
        z[i] = True
    else:
        z[i] = False
z = z.reshape((10, 5, 4))
z_reduce_all = tf.reduce_all(z, reduction_indices=[0, 2])

# tf.reduce_any
x = np.random.randint(0, 2, 10 * 5 * 4)
z = np.empty(10 * 5 * 4, dtype=np.bool)
for i, x_ in enumerate(x):
    if x_ > 0:
        z[i] = True
    else:
        z[i] = False
z = z.reshape((10, 5, 4))
z_reduce_any = tf.reduce_any(z, reduction_indices=[0, 2])

# tf.accumulate_n
inputs = [np.random.rand(5, 4, 3)] * 3
z_accumulate_n = tf.accumulate_n(inputs, shape=(10, 5, 4))

with tf.Session() as sess:

    print "tf.reduce_sum"
    print sess.run(z_reduce_sum)
    
    print "tf.reduce_prod"
    print sess.run(z_reduce_prod)
    
    print "tf.reduce_min"
    print sess.run(z_reduce_min)
    
    print "tf.reduce_max"
    print sess.run(z_reduce_max)
    
    print "tf.reduce_mean"
    print sess.run(z_reduce_mean)
    
    print "tf.reduce_all"
    print sess.run(z_reduce_all)
    
    print "tf.reduce_any"
    print sess.run(z_reduce_any)
    
    print "tf.accumulate_n"
    print sess.run(z_accumulate_n)
