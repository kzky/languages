#coding: utf-8


import urllib2
import re

fin_name = "/home/kzk/language/python/parse/man/fread.3.html"
pat = re.compile("<B>.*fread.*;")
fpin = file(fin_name)

for i in fpin:
    x = pat.search(i)
    if x is not None:
        print x.group()
        

fpin.close()

