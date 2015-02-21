#coding: utf-8

import urllib2

dat = urllib2.urlopen("https://graph.facebook.com/tomohiro.suguro")

for line in dat:
    print line
