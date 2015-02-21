#!/usr/bin/python
import numpy as np

# files
fname = "abalone.csv"
dirin = "/home/kzk/datasets/uci_origin/"
fin = dirin + fname
dirout = "/home/kzk/datasets/uci_csv/"
fout = dirout + fname

# read
dat = np.loadtxt(fin, delimiter=" ")

sex_category = set(dat[:, 0])

# convert
fpout = open(fout, "w")
for i in xrange(dat.shape[0]):
    line = str(int(dat[i, -1]))
    for s in sex_category:
        if dat[i, 0] == s:
            line = line + " 1"
        else:
            line = line + " 0"
    line = line + " " +   " ".join(map(str, dat[i, 1:-1])) + "\n"
    fpout.write(line)
fpout.close()


