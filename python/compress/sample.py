#!/usr/bin/python

import gzip
import bz2
import zipfile

## gzip
print "### gzip processing ###"
# write gzfile
fin = "/home/kzk/tmp/gzip_sample.gz"
fpout = gzip.open(fin, "wb")
fpout.write("test0\n")
fpout.write("test1\n")
fpout.write("test2")
fpout.close()

# read gzfile
fin = "/home/kzk/tmp/gzip_sample.gz"
fpin = gzip.open(fin, "r")
for line in  fpin:
    print line.strip()
fpin.close()

## bzip2
print "### bz2 processing ###"
# write bz2flie
fin = "/home/kzk/tmp/bz2_sample.bz2"
fpout = bz2.BZ2File(fin, "wb")
fpout.write("test0\n")
fpout.write("test1\n")
fpout.write("test2")
fpout.close()

# read bz2file
fin = "/home/kzk/tmp/bz2_sample.bz2"
fpin = bz2.BZ2File(fin, "r")
for line in  fpin:
    print line.strip()
fpin.close()

## zip
print "### zip processing ###"
fin = "/home/kzk/downloads/2753-2.zip"

zf = zipfile.ZipFile(fin, "r")
for f in zf.namelist():
    print f
    zf.read(f) ## reading
zf.close

