#coding: utf-8

import urllib2, urllib
import os
import os.path as path

up = urllib2.urlopen("http://python-tips.seesaa.net/article/148611391.html")

## the same
fp = file(path.join(path.abspath(""), "test.html"), "w")
for line in up:
    fp.write(line)
    
fp.close()
up.close();

## the same
urllib.urlretrieve("http://python-tips.seesaa.net/article/148611391.html", path.join(path.abspath(""), "test.html"))




