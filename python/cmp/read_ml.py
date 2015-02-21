#!/usr/bin/python

import time as t

fin = "/home/kzk/datasets/movie_lens/ml-10M100K/ratings.dat"

stime = t.time()
cnt = 0
for l in open(fin, "r"): 
    cnt += 1
    l.rstrip().split("::")
    if (cnt % 1000000 == 0): 
        print "progress", cnt

etime = t.time()

print etime - stime, "[s]"

