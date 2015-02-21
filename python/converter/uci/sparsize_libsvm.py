#!/usr/bin/python 
import sys
import time

#######################################
# Convert dense format to sparse one of libsvm format
## Origianl data format are as follows
## l<space>f1<space>f2<space>...<space>fn
## l<space>f1<space>f2<space>...<space>fn
## ...
## l<space>f1<space>f2<space>...<space>fn
## 
## Converted data format are as follows
## l<space>idx:v<space>idx:v<space>...<space>idx:v
## l<space>idx:v<space>idx:v<space>...<space>idx:v
## ...
## l<space>idx:v<space>idx:v<space>...<space>idx:v
#######################################

def print_usage():
    print """
python sparsize.py /home/kzk/datasets/uci_csv/adult.csv /home/kzk/datasets/uci_sparse/adult
"""

argv = sys.argv
fin = argv[1]
fout = argv[2]
delimiter1 = " " # for separator of values
delimiter2 = ":" # for index:value
if len(argv) >= 4:
    delimiter1 = argv[3]
    delimiter2 = argv[4]
    print_usage()
    
fout = fout + ".dat"
fpout = open(fout, "w")

st = time.time()
for l in open(fin, "r"):
   sl =  l.strip().split(delimiter1)
   out = sl[0] + delimiter1
   for i in xrange(1, len(sl)):
       if sl[i] != "0":
           out = out + str(i - 1) + delimiter2 + sl[i] + delimiter1
   fpout.write(out.strip())
   fpout.write("\n")

fpout.close()
et = time.time()
print "converting time: %s [s]" % (et - st)
