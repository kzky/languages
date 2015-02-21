#!/usr/bin/python
import random

fin = "/home/kzk/datasets/uci_csv/20news.csv"
fout = "/home/kzk/datasets/uci_tmp/20news.csv"

# read
dat = []
for l in open(fin):
    dat.append(l)

# randomize
fpout = open(fout, "w")
rdat = range(len(dat))
random.shuffle(rdat)
for i in rdat:
    fpout.write(dat[i])

fpout.close()

