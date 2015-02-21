#!/usr/bin/python

from StringIO import StringIO

output = ""
for l in open("/home/kzk/datasets/uci_csv/glass.csv"):
    output += l

fpout = open("/home/kzk/tmp/glass.csv", "w")
fpout.write(output)
fpout.close()
    
