import numpy as np
from scipy.linalg import pinv
from functools import reduce

import time
import multiprocessing as mp
from multiprocessing.pool import ThreadPool


class CPDALS(object):
    
    def __init__(self, ):
        self.pool = ThreadPool(mp.cpu_count() - 1)

    def solve(self, X, rank, max_iter=200, stopping_criterion=1e-5, 
            dtype=np.float32):
    
        N = X.ndim  # Tensor dimensions
        squared_norm_X = np.sum(X ** 2) # Frobenious norm square
        
        # Initialize
        A = [None for _ in range(N)]
        for n in range(1, N):
            A[n] = np.array(np.random.rand(X.shape[n], rank), dtype=dtype)

        # Solve ALS problem        
        criterion = 0
        for itr in range(max_iter):
            print(itr)
            criterion_prev = criterion
            
            # Fix one dimension
            for n in range(N):
                # Solve sub problem
                st = time.time()
                V = self.squared_hadamard_products(A, n)
                print("ElapsedTime(squared_hadamard_products)={}".format(time.time() - st))

                st = time.time()
                P = self.khatrirao_products(A, n)
                print("ElapsedTime(khatrirao_products)={}".format(time.time() - st))

                st = time.time()
                A_n = np.tensordot(
                    X, 
                    P, 
                    axes=(
                        [o for o in range(N) if o != n], 
                        np.arange(N-1)[::-1]))
                print("ElapsedTime(np.tensordot)={}".format(time.time() - st))

                st = time.time()
                A_n = A_n.dot(pinv(V))
                print("ElapsedTime(dot and pinv)={}".format(time.time() - st))

                # Normalize
                if itr == 0:
                    lmbda = np.sqrt((A_n ** 2).sum(axis=0))
                else:
                    lmbda = A_n.max(axis=0)
                    lmbda[lmbda < 1] = 1
                A[n] = A_n / lmbda

            # TODO: A bit difference between the reference and this criterion.
            st = time.time()
            X_approx = self.approximate_tensor(A, lmbda)
            squared_norm_residual = squared_norm_X + np.sum(X_approx**2) - 2 * np.sum(X*X_approx)   
            criterion = 1.0 - (squared_norm_residual / squared_norm_X)
            fit_change = abs(criterion_prev - criterion)
            print("ElapsedTime(Check Convergence)={}".format(time.time() - st))
            
            # Check convergence 
            if itr > 0 and fit_change < stopping_criterion:
                break
            
    
        return A, lmbda

    
    def khatrirao_product(self, X, Y):
        """Column-wise Katri-Rao product.
        
        Khatri-Rao product is the column-wise matching Kroncker product.
        For examples, suppose matrices X and Y each of which dimensions
        is (I, K) and (J, K).
        
        Khatri-Rao product between X and Y is defined by 
        
        .. math::
        
            X \odot Y = [a_1 \otimes b_1   a_2 \otimes b_2  \ldots a_K \otimes b_K ].
            
        
        Khatri-Rao product usually returns a matrix, or 2-dimensional array, 
        but in this function, it returns 3-dimensional array for the next use for
        tensor-dot product efficiently.
            
        Args:
            X (numpy.ndarray): the matrix which the number of the columns is the same as Y's. 
            Y (numpy.ndarray): the matrix which the number of the columns is the same as X's. 
        
        Returns:
            numpy.ndarray: the 3-dimensional array with the shape (I x J x K).
                 
        """
        
        n_rows = X.shape[0] * Y.shape[0]
        n_cols = X.shape[1]

        P = np.zeros((X.shape[0], Y.shape[0], n_cols))

        def outer_prod(X, Y, r, P):
            P[:, :, r] = np.outer(X[:, r], Y[:, r])
            return None

        future_list = []

        for r in range(n_cols):
            outer_product = np.outer(X[:, r], Y[:, r])  #TODO: dimension 
            n = np.prod(outer_product.shape)
            #P[:, r] = outer_product.reshape((n, ))
            P[:, :, r] = outer_product
        #     future = self.pool.apply_async(outer_prod, (X, Y, r, P))
        #     future_list.append(future)
        # for r in range(n_cols):
        #     future_list[r].get()
            
        return P
    
    def khatrirao_products(self, A, n, reverse=True):
        """Sequencial Khatri-Rao product without the `n`-th matirx in the revserse order.
        
        Args:
            A (list of numpy.ndarray): matrices where the number of columns for each matrix is are the same.
            n (int): N-th matrix which is ommited when computing Khatri-Rao product.
             
        Returns:
            numpy.ndarray: Result of the reduction of Khatri-Rao product for `A`. 
        """
        order = list(range(n)) + list(range(n + 1, len(A)))
        order = sorted(order, reverse=reverse)
        Z = reduce(lambda X, Y: self.khatrirao_product(X, Y), 
                   [A[o] for o in order if o != n])    
        return Z
    
    def squared_hadamard_products(self, A, n):
        """Sequencial squared Hadamard product without the `n`-th matirx.
        
        Args:
            A (list of numpy.ndarray): matrices where the number of columns for each matrix is are the same.
            n (int): N-th matrix which is ommited when computing Khatri-Rao product.
            
        """
        order = list(range(n)) + list(range(n + 1, len(A)))
        V = 1.
        for o in order:
            V = V * np.dot(A[o].T, A[o])
        return V
    
    def approximate_tensor(self, A, lmbda):
        """Compute the approximate original tensor
        
        Args:
            A (list of numpy.ndarray): matrices where the number of columns for each matrix is are the same.
            lmbda (numpy.ndarray): 1-dimensional np.ndarray
        """
        rank = len(lmbda)
        Z = np.zeros([a.shape[0] for a in A])

        def outer_prod(A, lmbda, r):
            a0 = A[0][:, r]
            for a in A[1:]:
                a0 = np.multiply.outer(a0, a[:, r])
            return lmbda[r] * a0

        future_list = []
        for r in range(rank):
            # a0 = A[0][:, r]
            # for a in A[1:]:
            #     a0 = np.multiply.outer(a0, a[:, r])
            # Z[...] += lmbda[r] * a0
            future = self.pool.apply_async(outer_prod, (A, lmbda, r))
            future_list.append(future)

        results = []
        for r in range(rank):
            results.append(future_list[r].get())
        Z = reduce(lambda X, Y: X + Y, results)
            
        return Z

