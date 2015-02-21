#coding: utf-8

import os

print os.path.abspath("")


files = os.listdir("/home/kzk")

for f in files:
    if f[0:1] != ".":
        print f
