#!/usr/bin/python

# http://docs.python.jp/2.6/library/re.html
import re

pat = re.compile("test[0-9]+")
print pat.match("h test f") # match from the begining

# only first matched string and return macher obj
mat = pat.search("h test0 f test test1, test23")
print mat.group()
print mat.group(0)
try:
    print mat.group(1)
except Exception as e:
    print e

# find all matched string and return as list
list = pat.findall("h test0 f test test1, test23")
print list
