#!/usr/bin/env python

import tensorflow as tf
import numpy as np


# tf.diag
x = np.random.rand(10)
z_diag = tf.diag(x)

# tf.transpose
x = np.random.rand(5, 2, 3)
z_transpose = tf.transpose(x, perm=[1, 2, 0])

# tf.matmul
x = np.random.rand(2, 3)
y = np.random.rand(3, 2)
z_matmul = tf.matmul(x, y)

# tf.batch_matmul
x = np.random.rand(10, 2, 3)
y = np.random.rand(10, 3, 2)
z_batch_matmul = tf.batch_matmul(x, y)

# tf.matrix_determinant
x = np.random.rand(5, 5)
z_matrix_determinant = tf.matrix_determinant(x)

# tf.batch_matrix_determinant
batch_x = np.random.rand(10, 5, 5)
z_batch_matrix_determinant = tf.batch_matrix_determinant(batch_x)

# tf.matrix_inverse
x = np.random.rand(10, 10)
z_matrix_inverse = tf.matrix_inverse(x)

# tf.batch_matrix_inverse
batch_x = np.random.rand(10, 5, 5)
z_batch_matrix_inverse = tf.batch_matrix_inverse(batch_x)

# tf.cholesky
x = np.random.rand(10, 10)
z_cholesky = tf.cholesky(x)

# tf.batch_cholesky
batch_x = np.random.rand(10, 5, 5)
z_batch_cholesky = tf.batch_cholesky(x)

# tf.self_adjoint_eig
x = np.random.rand(10, 8)
z_self_adjoint_eig = tf.self_adjoint_eig(x)

# tf.batch_self_adjoint_eig
batch_x = np.random.rand(10, 8, 5)
z_batch_self_adjoint_eig = tf.batch_self_adjoint_eig(batch_x)

with tf.Session() as sess:
    print "tf.diag"
    print sess.run(z_diag)
    print "tf.transpose"
    print sess.run(z_transpose)
    print "tf.matmul"
    print sess.run(z_matmul)
    print "tf.batch_matmul"
    print sess.run(z_batch_matmul)
    print "tf.matrix_determinant"
    print sess.run(z_matrix_determinant)
    print "tf.batch_matrix_determinant"
    print sess.run(z_batch_matrix_determinant)
    print "tf.matrix_inverse"
    print sess.run(z_matrix_inverse)
    print "tf.batch_matrix_inverse"
    print sess.run(z_batch_matrix_inverse)
    print "tf.cholesky"
    print sess.run(z_cholesky)
    print "tf.batch_cholesky"
    print sess.run(z_batch_cholesky)
    print "tf.self_adjoint_eig"
    print sess.run(z_self_adjoint_eig)
    print "tf.batch_self_adjoint_eig"
    print sess.run(z_batch_self_adjoint_eig)
