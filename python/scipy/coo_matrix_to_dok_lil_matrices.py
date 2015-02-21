#!/usr/bin/python

from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import dok_matrix
import time

# vars
fin = "/home/kzk/datasets/uci_tmp/20news/20news-bydate/matlab/train.data"

## datea format
# i, j, v
# e.g., 
# 1 1 4
# 1 2 2
# 1 3 10
# 1 4 4
# 1 5 2
# 1 6 1
# 1 7 1
# 1 8 1
# 1 9 3

row = []
col = []
data = []

# load starts
print "load starts"
st = time.time()
for l in open(fin):
    sl = l.strip().split(" ")
    row.append(int(sl[0]))
    col.append(int(sl[1]))
    data.append(float(sl[2]))

et = time.time()
print "load finished"
print "time: %f [s]" % (et - st)

# create coo_matrix
print "create coo_matrix"
st = time.time()
a = coo_matrix((data, (row, col)))
et = time.time()
print "time: create coo_matrix: %f [s]" % (et - st)
print "dim = ", a.shape
print type(a)

# multiply (n by m) x (m by n)
print "multiply (n by m) x (m by n) starts"
st = time.time()
a.dot(a.T)
et = time.time()
print "multiply (n by m) x (m by n) finished"
print "time: multiply (n by m) x (m by n): %f [s]" % (et - st)
 
# multiply (m by n) x (n by m)
print "multiply (n by m) x (m by n) starts"
st = time.time()
a.dot(a.T)
et = time.time()
print "multiply (n by m) x (m by n) finished"
print "time: multiply (m by n) x (n by m): %f [s]" % (et - st)

# create dok_matrix
print "create dok_matrix"
st = time.time()
b = a.todok()
et = time.time()
print "time: create dok_matrix: %f [s]" % (et - st)
print type(b)

# multiply (n by m) x (m by n)
print "multiply (n by m) x (m by n) starts"
st = time.time()
b.dot(b.T)
et = time.time()
print "multiply (n by m) x (m by n) finished"
print "time: multiply (n by m) x (m by n): %f [s]" % (et - st)

# multiply (m by n) x (n by m)
print "multiply (n by m) x (m by n) starts"
st = time.time()
b.dot(b.T)
et = time.time()
print "multiply (n by m) x (m by n) finshed"
print "time: multiply (m by n) x (n by m): %f [s]" % (et - st)

# create lil_matrix
print "create lil_matrix"
st = time.time()
b = a.tolil()
et = time.time()
print "time: create lil_matrix: %f [s]" % (et - st)
print type(b)

# multiply (n by m) x (m by n)
print "multiply (n by m) x (m by n) starts"
st = time.time()
b.dot(b.T)
et = time.time()
print "multiply (n by m) x (m by n) finished"
print "time: multiply (n by m) x (m by n): %f [s]" % (et - st)

# multiply (m by n) x (n by m)
print "multiply (n by m) x (m by n) starts"
st = time.time()
b.dot(b.T)
et = time.time()
print "multiply (n by m) x (m by n) finshed"
print "time: multiply (m by n) x (n by m): %f [s]" % (et - st)
