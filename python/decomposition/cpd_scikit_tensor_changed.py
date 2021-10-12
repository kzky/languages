from sktensor import dtensor, cp_als

import numpy as np
from cpd_changed import CPDALS 
import time

def main():
    # Settings
    outmap = 128
    inmap = 64
    k = 3
    seed = 412
    np.random.seed(seed)
    oik = np.random.rand(outmap, inmap, k**2)
    X = oik
    rank = 32
    max_iter = 200
    stopping_criterion = 1e-5
    dtype = np.float32
    
    # Canonical Polyadec Decompostion by ALS
    cpdals = CPDALS()
    st = time.time()
    A, lmbda = cpdals.solve(X=X, rank=rank, max_iter=max_iter, dtype=dtype)
    print("ElapsedTime: {}".format(time.time() - st))
              
    #print(A[0])
    #print(lmbda)
    
    # Canonical Polyadec Decompostion by ALS (original)
    np.random.seed(seed)
    oik = np.random.rand(outmap, inmap, k**2)

    X = dtensor(oik)
    st = time.time()
    ret_origin, _, _, _ = cp_als(X, rank=rank, init='random', max_iter=max_iter, dtype=dtype)
    print("ElapsedTime: {}".format(time.time() - st))

    #print(ret_origin.U[0])
    #print(ret_origin.lmbda)
    
    
    for atol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        # Comparison for factorized matrices
        print("atol={}".format(atol))
        for u0, u1 in zip(A, ret_origin.U):
            print(np.allclose(u0, u1,  rtol=1e-04, atol=atol))
#             print(u0[0])
#             print(u1[0])
        
        # Comparison for lambda
        print(np.allclose(lmbda, ret_origin.lmbda, rtol=1e-04, atol=atol))
#         print(lmbda, ret_origin.lmbda)
                
if __name__ == '__main__':
    main()
