#!/usr/bin/env python

"""
Not meet my demand
"""

import numpy as np

from scipy.sparse import coo_matrix
from sklearn.datasets import load_svmlight_file
def reshape(X, shape):
    """Reshape the sparse matrix `a`.

    Returns a csr_matrix with shape `shape`.
    """
    X = X.tolil().reshape(shape).tocsr()
    return X


(X_l, y_l) = load_svmlight_file("/home/kzk/datasets/sparse_news20_ssl_lrate_fixed_1_30_1_98/news20/1_l.csv")
(X_u, y_u) = load_svmlight_file("/home/kzk/datasets/sparse_news20_ssl_lrate_fixed_1_30_1_98/news20/1_u.csv")

print "shape of X_l"
print X_l.shape

print "shape of X_u"
print X_u.shape


ds = [X_l.shape[1], X_u.shape[1]]
idx = np.argmax(ds)
d = ds[idx]
print "dimension"
print d
if idx == 0:
    X_u = reshape(X_u, (X_u.shape[0], d))
else:
    X_l = reshape(X_l, (X_l.shape[0], d))

print X_u
print X_u.shape




