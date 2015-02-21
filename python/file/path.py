#!/usr/bin/python

import os.path as path

pathname = "/home/kzk/downloads/yasnippet-snippets-master/python-mode"

print "abspath:", path.abspath(pathname)
print "basename:", path.basename(pathname)
print "dirname:", path.dirname(pathname)
print "exists:", path.exists(pathname)
print "split:", path.split(pathname)
print "getmtime:", path.getmtime(pathname)
print "getatiem:", path.getatime(pathname)
print "getctime:", path.getctime(pathname)





