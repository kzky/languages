#coding: utf-8

import string

fp = file("/home/kzk/.zshrc", "r")

for line in fp:
    print line

fp.close()

fp = file("/home/kzk/test.txt", "a")

lis = ["1", "3", "4"]

for i in lis:
    fp.write("test\n")
    
fp.close()

