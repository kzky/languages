#!/usr/bin/python 
import sys
import time

#######################################
# Convert dense format to sparse one of coo_matrix format
## Origianl data format are as follows
## l<space>f1<space>f2<space>...<space>fn
## l<space>f1<space>f2<space>...<space>fn
## ...
## l<space>f1<space>f2<space>...<space>fn
## 
## Converted data format for samples are as follows
## i<space>j<space>v
## i<space>j<space>v
## ...
## i<space>j<space>v
## 
## Converted data format for labels are as follows
## l
## l
## ...
## l
#######################################

def print_usage():
    print """
python sparsize.py /home/kzk/datasets/uci_csv/adult.csv /home/kzk/datasets/uci_sparse/adult
"""

argv = sys.argv
fin = argv[1]
fout = argv[2]
delimiter = " " # for separator of values
if len(argv) >= 4:
    delimiter = argv[3]
    print_usage()
    
fout_sample = fout + ".sample"
fout_label = fout + ".label"
fpout_sample = open(fout_sample, "w")
fpout_label = open(fout_label, "w")

st = time.time()
cnt = 0
for l in open(fin, "r"):
   sl =  l.strip().split(delimiter)
   # label
   fpout_label.write(sl[0])
   fpout_label.write("\n")
   # sample
   out = ""
   for i in xrange(1, len(sl)):
       if sl[i] != "0":
           out = str(cnt) + delimiter + str(i - 1) + delimiter + sl[i]
           fpout_sample.write(out)
           fpout_sample.write("\n")
   cnt += 1

fpout_sample.close()
fpout_label.close()
et = time.time()
print "converting time: %s [s]" % (et - st)
