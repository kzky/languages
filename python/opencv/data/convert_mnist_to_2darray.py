#!/usr/bin/env python

import cPickle
import gzip
import pickle as pkl

# A sample represents 1d-array
## *_set[0] is 2d-array,  (n, d), *_set[1] is 1d-array

input_filepath = "/home/kzk/downloads/mnist.pkl.gz"
output_dirpath = "/home/kzk/datasets/mnist/mnist_0-1"

# Load the dataset
f = gzip.open(input_filepath, 'rb')
train_set, valid_set, test_set = cPickle.load(f)  
f.close()

# train_set
train_set_reshaped = []
for train in train_set[0]:
    train_set_reshaped.append(train.reshape(28, 28))
    pass
train_set = [train_set_reshaped, train_set[1]]
output_filepath = "{}/{}".format(output_dirpath, "train_set.pkl")
with open(output_filepath, "w") as fpout:
    pkl.dump(train_set, fpout)
    pass

# valid_set
valid_set_reshaped = []
for valid in valid_set[0]:
    valid_set_reshaped.append(valid.reshape(28, 28))
    pass
valid_set = [valid_set_reshaped, valid_set[1]]
output_filepath = "{}/{}".format(output_dirpath, "valid_set.pkl")
with open(output_filepath, "w") as fpout:
    pkl.dump(valid_set, fpout)
    pass

# test_set
test_set_reshaped = []
for test in test_set[0]:
    test_set_reshaped.append(test.reshape(28, 28))
    pass
test_set = [test_set_reshaped, test_set[1]]
output_filepath = "{}/{}".format(output_dirpath, "test_set.pkl")
with open(output_filepath, "w") as fpout:
    pkl.dump(test_set, fpout)
    pass


