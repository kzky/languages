#coding: utf-8


import re

fp = file("/home/kzk/language/python/parse/man/dir_section_2.html")

pat = re.compile("<a.*>(.*)\(2\)</a>")
for line in fp:
    x = pat.search(line)
    if x is not None:
        print x.group(1)
fp.close()

##
##fp2 = file("/home/kzk/language/python/parse/man/dir_section_3.html")
##pat = re.compile("<a href\=\.(.*\.html)>.*</a>")
##for line in fp2:
##    x = pat.search(line)
##    if x is not None:
##        print x.group(1)
##
##
##fp2.close()        
##
    

