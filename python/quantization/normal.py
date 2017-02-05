import numpy as np
import time
import scipy.stats

def main():
    # Settings
    loc= 0.0
    scale = 0.001
    size = 10000
    p = 0.05
    g = np.random.normal(loc=loc, scale=scale, size=size)
    q = scipy.stats.norm.ppf(q=p, loc=loc, scale=scale)

    print("Percent Point: {}".format(q))
    print("Sigma/2/p: {}".format(np.std(g) / 2 / p))


if __name__ == '__main__':
    main()
