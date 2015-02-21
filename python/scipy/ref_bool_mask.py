import scipy as sp
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand

fin = "/home/kzk/datasets/uci_csv/iris.csv"
data = np.loadtxt(fin)
labels = data[:, 0]
data_masked = data[(labels == 0) | (labels == 1) | (labels == 2), :]

print "### different id for obj level ###"
print id(data), id(data_masked)
print id(data) == id(data_masked)

print "### same id for row level ###"
print id(data[60,:]), id(data_masked[60, :])
print id(data[60, :]) == id(data_masked[60, :]) 

print "### same id for column level ###"
print id(data[:, 2]), id(data_masked[:, 2])
print id(data[:, 2]) == id(data_masked[:, 2]) 

print "### change values of the last row ###"
lrow = data.shape[0] - 1
data[lrow, 2] = 10000
print "data", data[lrow, :]
print "data_masked", data_masked[lrow, :]

print "The same reference is not returned using boolean mask!"
print "The same reference are returned in the case of sciling, data[j, :] ,data[:, j]"




