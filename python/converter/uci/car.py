#!/usr/bin/python
import numpy as np
from collections import defaultdict

# files
fname = "car.csv"
dirin = "/home/kzk/datasets/uci_origin/"
fin = dirin + fname
dirout = "/home/kzk/datasets/uci_csv/"
fout = dirout + fname

# read
dat = np.loadtxt(fin, delimiter=" ")
categoris = defaultdict()
categoris[1] = set(dat[:, 1]) 
categoris[2] = set(dat[:, 2]) 
categoris[3] = set(dat[:, 3]) 
categoris[4] = set(dat[:, 4]) 
categoris[5] = set(dat[:, 5]) 
categoris[6] = set(dat[:, 6]) 
ckeys = categoris.keys()

# convert
fpout = open(fout, "w")
for i in xrange(dat.shape[0]):
    l = str(int(dat[i, 0]))
    for j in xrange(1, dat.shape[1]):
        if (j in ckeys):  # expand
            for s in categoris[j]:
                if dat[i, j] == s:
                    l = l + " 1"
                else:
                    l = l + " 0"
        else:
            l = l + " " + str(dat[i, j])
    l = l + "\n"
    fpout.write(l)
fpout.close()


