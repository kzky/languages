import scipy as sp
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand

## construct sparse matrix and set values
A = lil_matrix((1000, 1000))
A[0, :100] = rand(100)
A[1, 100:200] = A[0, :100]
A.setdiag(rand(1000))

## convert to csr and solve Ax = b
A = A.tocsr() 
b = rand(1000)
x = spsolve(A, b) ## with csc, csr only

## compare the dense matrix solutoin
x_ = solve(A.todense(), b)
err = norm(x-x_)
err < 1e-10

## transepose of multiple matrix format
data = rand(10, 10)
lil_mat = lil_matrix((10, 10))
lil_mat[:, : ] = data
# sp.transpose(lil_mat) # x
# sp.transpose(lil_mat.tobsr()) # x
sp.transpose(lil_mat.tocoo())
sp.transpose(lil_mat.todok())
sp.transpose(lil_mat.tocsc())
sp.transpose(lil_mat.tocsr())



