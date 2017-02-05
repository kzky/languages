
import numpy as np
import time
import scipy.stats

def main():
    # Settings
    loc= 0.0
    scale = 0.001
    size = 10000

    g = np.random.normal(loc=loc, scale=scale, size=size)

    idx = np.where(g > 0)[0]
    mean_point = np.mean(g[idx])
    p = scipy.stats.norm.cdf(mean_point, loc=loc, scale=scale)
    print("Scale:{},Mean:{},Percent:{}".format(scale, mean_point, 1. - p))


if __name__ == '__main__':
    main()
