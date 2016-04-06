#!/usr/bin/env python

import numpy as np

print "Binary Operation with Numpy\n"
a = np.random.uniform(0, 10, 10).astype(int)
b = np.random.uniform(0, 10, 10).astype(int)

print a, "\n"
print b, "\n"

print "a << b"
print a << b, "\n"

print "a << 2"
print a << 2, "\n"


