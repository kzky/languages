import time
import numpy as np
import scipy.stats


def main():
    loc = 0
    scale = 0.003
    size = int(100 * 1e4)
    g = scipy.stats.laplace.rvs(loc=0., scale=scale, size=size)
    #p_g = scipy.stats.laplace.ppf(loc=0., scale=scale, size=1000000)
    
    log_abs_g_one = np.log2(np.abs(g) + 1)
    min_log_abs_g_one = np.min(log_abs_g_one)
    max_log_abs_g_one = np.max(log_abs_g_one)
    q = 127 * (log_abs_g_one - min_log_abs_g_one) / (max_log_abs_g_one - min_log_abs_g_one)

    # Sign
    st = time.time()
    q = (q * np.sign(g)).astype(np.int8)
    et = time.time()
    print("Method:Sign,ElapsedTime:{}[s]".format(et - st))

    # Where0
    st = time.time()
    neg_idx = np.where(g < 0)[0]
    q[neg_idx] = (-1. * q[neg_idx]).astype(np.int8)
    et = time.time()
    print("Method:Where(Neg),ElapsedTime:{}[s]".format(et - st))

    # Where1
    st = time.time()
    pos_idx = np.where(g > 0)[0]
    neg_idx = np.where(g < 0)[0]
    q[neg_idx] = (-1. * q[neg_idx]).astype(np.int8)
    et = time.time()
    print("Method:Where(Pos+Neg),ElapsedTime:{}[s]".format(et - st))

    print("Num of 0 is {}/{}".format()len(np.where(np.abs(q) == 0)[0]), len(q))
if __name__ == '__main__':
    main()
    
