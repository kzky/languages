import numpy as np
import time
import scipy.stats

def main():
    size = int(10 * 1e6)
    x = np.random.randn(size)
    print("Speed depends on sparsity")
    threshold = np.abs(scipy.stats.norm.ppf(0.025))  

    # Sparsize
    st = time.time()
    idx_pos = np.where(x > threshold)[0]
    idx_neg = np.where(x < -threshold)[0]
    et = time.time()
    print("Sparsize:{}[s]".format(et - st))

    # Sparsize (Sign)
    st = time.time()
    idx_abs = np.where(np.abs(x) > threshold)[0]
    idx = idx_abs * np.sign(x[idx_abs]).astype(np.int32)  # Take care the order
    et = time.time()
    print("Sparsize(Sign):{}[s]".format(et - st))
          
if __name__ == '__main__':
    main()

