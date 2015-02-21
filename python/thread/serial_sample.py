#!/usr/bin/python
# -*- coding: utf-8 -*-

# url
## http://ja.pymotw.com/2/threading/
## http://docs.python.jp/2.7/library/threading.html

##############
## zip fileを解凍する
##############

import threading
import glob
import time
import zipfile

path = "/home/kzk/downloads/*.zip"
zipfiles = glob.glob(path)

## reading onley unzip
def unzip(zfin):
    zf = zipfile.ZipFile(zfin, "r")
    for f in zf.namelist():  # zipは複数ファイルが１つにまとめられている前提のため
        #print "unzip", f
        zf.read(f)  # reading only
    zf.close

## non-thread for comparison
st = time.time()
for zf in zipfiles:
    unzip(zf)
    
et = time.time()
print "total execution time without threading: ", (et - st), "[s]"

# total execution time without threading:  57.6119029522 [s]
