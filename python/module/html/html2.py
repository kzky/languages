#coding:utf-8

import urllib2
from htmllib import HTMLParser
from formatter import NullFormatter
import os
import re

url_name = "http://b.hatena.ne.jp/hotentry"
html_data = urllib2.urlopen(url_name)
parser = HTMLParser(NullFormatter())

try:
    parser.feed(html_data.read())
except TypeError:
    print "type error"

pat = re.compile("^http.*")
for link in parser.anchorlist:
    x = pat.search(link)
    if x is not None:
        print x.group(0)

