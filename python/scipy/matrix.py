import scipy as sp
from numpy.linalg import solve, norm
from numpy.random import rand

## matrix product test
mat = sp.array([[1,2], [3,4]])
print mat
print mat.dot(mat[:,0])
print mat[0, :].dot(mat) ## always converted to column vector



