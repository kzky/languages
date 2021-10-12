from sktensor import dtensor, cp_als

import numpy as np

from cpd import cpd_als

import time

def main():
    # Settings
    outmap = 128
    inmap = 128
    k = 3
    seed = 412
    np.random.seed(seed)
    oik = np.random.rand(outmap, inmap, k**2)
    X = dtensor(oik)
    
    rank = 287
    max_iter = 50
    stopping_criterion = 1e-5
    
    # Canonical Polyadec Decompostion by ALS
    st = time.time()
    ret = cpd_als(X=X, rank=rank, max_iter=max_iter)
    print("ElapsedTime[s]: {}".format(time.time() - st))
      
#     print(ret.U[0])
#     print(ret.lmbda)
    
    # Canonical Polyadec Decompostion by ALS (original)
    np.random.seed(seed)
    oik = np.random.rand(outmap, inmap, k**2)

    X = dtensor(oik)
    st = time.time()
    ret_origin, _, _, _ = cp_als(X, rank=rank, init='random')
#     print(ret_origin.U[0])
#     print(ret_origin.lmbda)
    
     
    # Comparison
    for u0, u1 in zip(ret.U, ret_origin.U):
        print(np.allclose(u0, u1,  rtol=1e-05, atol=1e-04,))
        
                
if __name__ == '__main__':
    main()
