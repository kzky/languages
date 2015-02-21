#!/usr/bin/python
import numpy as np
from collections import defaultdict

# files
fname = "glass.csv"
dirin = "/home/kzk/datasets/uci_origin/"
fin = dirin + fname
dirout = "/home/kzk/datasets/uci_csv/"
fout = dirout + fname

# read
dat = np.loadtxt(fin, delimiter=" ")
# convert
fpout = open(fout, "w")
for i in xrange(dat.shape[0]):
    # l = str(int(dat[i, 0])) + " " + " ".join(map(str, dat[i, 2:])) + "\n" # no need?
    l = str(int(dat[i, 0])) + " " + " ".join(map(str, dat[i, 1:])) + "\n"
    fpout.write(l)
fpout.close()


