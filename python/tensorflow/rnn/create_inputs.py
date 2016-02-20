#!/usr/bin/env python

import numpy as np
import tensorflow as tf

"""
TensorFlow rnn library requires inputs, list of tensor of size, (batch_size x vector_length). When looking over the time axis, vectors over the batch axis must be aligned by time, so that the easy way to create list of tensor is, first create 3-rank tensor of size (batch_size, time_length, vector_length), second correctly reshape that to (time_length, batch_size, vector_length), then apply unpack.
"""

# Suppose if we have long sequence and create non-overlapping sequence of vector
N = 10000
x = np.arange(N)

# First,  just reshape
batch_size = 10
time_length = 100
vector_length = 10
X = np.reshape(x, (batch_size, time_length, vector_length))

# Second,  reshape correctly
Y = np.zeros((time_length, batch_size, vector_length))

for i in xrange(batch_size):
    for j in xrange(time_length):
        Y[j, i, :] = X[i, j, :]
        
# Finally, unpack
Z = list(Y)

# Confirm that in the same batch, vectors are correctly aligned
print Z[0][0, :]
print Z[1][0, :]

print Z[12][4, :]
print Z[13][4, :]
