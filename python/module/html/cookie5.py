#coding: utf-8

from urllib2 import urlopen
import urllib
import os
import re
import htmllib
import httplib

url_name = "https://grp03.id.rakuten.co.jp"
con = httplib.HTTPSConnection(url_name, 443)
con.request("GET", "/rms/nid/personalfwd?profile_id=326499073&labelId=1&service_id=p06")
res = con.response()

print "status"
print res.status, res.reason
print "header"
print res.headers()
print "msg"
print res.msg


