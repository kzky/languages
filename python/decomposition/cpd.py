import numpy as np
from scipy.linalg import pinv

from sktensor.ktensor import ktensor
from sktensor.core import norm, khatrirao

def cpd_als(X, rank, max_iter=200, stopping_criterion=1e-5, 
        dtype=np.float32):
    """

    """

    # Initialize options

    N = X.ndim
    normX = norm(X)  # Frobenious norm  

    U = _init(X, N, rank, dtype)
    criterion = 0
    
    for itr in range(max_iter):
        criterion_prev = criterion
        print(itr)
        # Fix one dimension (frontal slice)
        for n in range(N):
            Y = np.ones((rank, rank), dtype=dtype)
            
            # Use the other frontal slices
            for i in (list(range(n)) + list(range(n + 1, N))):
                Y = Y * np.dot(U[i].T, U[i])

            Unew = X.uttkrp(U, n)
            Unew = Unew.dot(pinv(Y))

            # Normalize
            if itr == 0:
                lmbda = np.sqrt((Unew ** 2).sum(axis=0))
            else:
                lmbda = Unew.max(axis=0)
                lmbda[lmbda < 1] = 1
            U[n] = Unew / lmbda
        
        P = ktensor(U, lmbda)

        #TODO: ktensor function: norm and innerprod
        normresidual = normX ** 2 + P.norm() ** 2 - 2 * P.innerprod(X)
        criterion = 1 - (normresidual / normX ** 2)
        fitchange = abs(criterion_prev - criterion)

        if itr > 0 and fitchange < stopping_criterion:
            break

    return P


def _init(X, N, rank, dtype):
    """
    Initialization for CP models
    """
    Uinit = [None for _ in range(N)]
    for n in range(1, N):
        Uinit[n] = np.array(np.random.rand(X.shape[n], rank), dtype=dtype)
    return Uinit



