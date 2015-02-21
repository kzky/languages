#!/usr/bin/python

fin = "/home/kzk/datasets/uci_csv/news20.csv"

cnt = 0
for l in open(fin):
    sl = l.strip().split(" ")
    if cnt % 10000 == 0:
        print cnt
        #print sl
    cnt += 1
    
