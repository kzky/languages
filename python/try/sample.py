#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

print "args are", sys.argv[:]

for arg in sys.argv[1:]:
    try:
        f = open(arg, 'r')
    except IOError as e:
        print 'cannot open', arg
        print e
    else:
        print arg, 'has', len(f.readlines()), 'lines'
        f.close()
