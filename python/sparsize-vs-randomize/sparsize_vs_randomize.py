import numpy as np
import time
import scipy.stats

def sparsize(x, p):
    p = p / 2
    q = np.abs(scipy.stats.norm.ppf(p))
    pos_idx = np.where(x > q)[0]
    neg_idx = np.where(x < -q)[0]
    return np.concatenate((pos_idx, neg_idx))

def randomize(x, p, size):
    n = int(size * p)
    idx = np.random.choice(size, n, replace=False)
    grad = x[idx]
    return grad

def main():
    size = int(20 * 1e6)
    p = 0.07
    x = np.random.randn(size)

    st = time.time()
    sparsize(x, p)
    et = time.time() - st
    print("Sparsize:{}[s]".format(et))

    st = time.time()
    randomize(x, p, size)
    et = time.time() - st
    print("Randomize:{}[s]".format(et))

if __name__ == '__main__':
    main()
    
