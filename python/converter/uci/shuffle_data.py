#!/usr/bin/env python

import time
import os
import glob
import csv
import numpy as np

from collections import defaultdict

user = "kzk"
src_path = "/home/%s/datasets/news20/news20.dat" % user
dst_train_path = "/home/%s/datasets/news20/news20.tr.dat" % user
dst_test_path = "/home/%s/datasets/news20/news20.t.dat" % user

indexed_data = defaultdict()
st  = time.time()
# read
with open(src_path) as fpin:
    reader = csv.reader(fpin, delimiter=" ")
    for i, r in enumerate(reader):
        indexed_data[i] = r
    pass
et  = time.time()
print "read time: %f [s]" % (et - st)

# shuffle
st  = time.time()
keys = indexed_data.keys()
np.random.shuffle(keys)
et  = time.time()
print "shuffle time: %f [s]" % (et - st)

# write
st  = time.time()
len_index = len(indexed_data)
## traning files
with open(dst_train_path, "w") as fpout:
    reader = csv.writer(fpout, delimiter=" ")
    for row in indexed_data.values()[0:len_index / 2]:
        reader.writerow(row)
        pass
    pass
with open(dst_test_path, "w") as fpout:
    reader = csv.writer(fpout, delimiter=" ")
    for row in indexed_data.values()[len_index / 2:len_index]:
        reader.writerow(row)
        pass
    pass
et  = time.time()
print "write time: %f [s]" % (et - st)

