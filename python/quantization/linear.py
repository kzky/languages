import time
import numpy as np
import scipy.stats


def main():
    loc = 0
    scale = 0.003
    size = int(100 * 1e4)
    g = scipy.stats.laplace.rvs(loc=0., scale=scale, size=size)
    #p_g = scipy.stats.laplace.ppf(loc=0., scale=scale, size=1000000)
    
    abs_g_one = np.abs(g)
    min_abs_g_one = np.min(abs_g_one)
    max_abs_g_one = np.max(abs_g_one)
    q = 127 * (abs_g_one - min_abs_g_one) / (max_abs_g_one - min_abs_g_one)

    # Sign
    st = time.time()
    q = (q * np.sign(g)).astype(np.int8)
    et = time.time()
    print("Num of 0 is {}/{}".format()len(np.where(np.abs(q) == 0)[0]), len(q))    

if __name__ == '__main__':
    main()
    
