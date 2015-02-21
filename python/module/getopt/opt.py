#coding: utf-8

import getopt
import sys

(opts, args) = getopt.getopt(sys.argv[1:], "abc:d:")



for (x, y) in opts:
    print "%s, %s" % (x, y)


