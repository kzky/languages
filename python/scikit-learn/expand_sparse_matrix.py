import time
import numpy as np

from sklearn.datasets import load_svmlight_file
from scipy import sparse
from scipy.sparse import csr_matrix

# http://stackoverflow.com/questions/17386835/extending-an-existing-matrix-in-scipy

filename = "/home/kzk/datasets/news20/news20.dat"


st = time.time()
(X, y) = load_svmlight_file(filename)
et = time.time()
print "ellapsed time: %f [s]" % (et - st)
#print y
#print X

print type(y)
print type(X)
print X.shape
print X.ndim


# add bias
f_dims = X.shape[1]
x = X[1001, :]
one = csr_matrix(([1], ([0], [0])))
print x
print sparse.hstack([x, one])

print x.indptr
print x.indices
print x._shape
print x.data




#x.data = np.hstack((x.data, 1))
#x.indices = np.hstack((x.indices, f_dims+1))
#x.indptr[1] = 15
#x._shape = (1, f_dims + 1)

