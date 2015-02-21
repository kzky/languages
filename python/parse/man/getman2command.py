#coding: utf-8

#library
import urllib2
import re
import time
import os

## input url
fpin = file("/home/kzk/language/python/parse/man/sec2.list", "r")
uri_header = "http://man7.org/linux/man-pages/man2/"
suffix = ".2.html"

## output file
foutname = "/home/kzk/dataset/man/man2.cmd"

if os.path.isfile(foutname):
    os.remove(foutname)
    fpout = file(foutname, "w")
else:
    fpout = file(foutname, "a")

## scraping
for i in fpin: ## for url
    uriname = i[0:len(i)-1]
    urlname = uri_header + uriname +  suffix

    ## get html data
    time.sleep(1)
    print urlname
    data = urllib2.urlopen(urlname)
    
    ## regular expression
    regexp = "<B>.*" + uriname + ".*<B>\);"
    pat = re.compile(regexp)
    
    for line in data:
        cmd = pat.search(line)

        if cmd is not None:
            fpout.write(cmd.group() + "\n")

fpout.close()
fpin.close()
