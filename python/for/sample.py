#!/usr/bin/python
# -*- coding: utf-8 -*-

# for loop
for i in range(0, 9):
    print i

# for loop (as usual, but system error)
#for(i=0; i < 10; i++):
#    print i

# list内包表記
list = [i for i in range(0, 9)]
print list

list2 = [i for i in range(0, 9) if i % 2 == 0]
print list2

# xrange
for i in reversed(xrange(0, 10000, 2)):
    print i

    
