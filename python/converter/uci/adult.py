#!/usr/bin/python
import numpy as np
from collections import defaultdict

# files
fname = "adult.csv"
dirin = "/home/kzk/datasets/uci_origin/"
fin = dirin + fname
dirout = "/home/kzk/datasets/uci_csv/"
fout = dirout + fname

# read
dat = np.loadtxt(fin, delimiter=" ")
categoris = defaultdict()
categoris[2] = set(dat[:, 2])
categoris[4] = set(dat[:, 4])
categoris[6] = set(dat[:, 6])
categoris[7] = set(dat[:, 7])
categoris[8] = set(dat[:, 8])
categoris[9] = set(dat[:, 9])
categoris[10] = set(dat[:, 10])
categoris[14] = set(dat[:, 14])
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


