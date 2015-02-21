#coding: utf-8

x = [1, 4]
y = [1, 4]

print type(x)

if type(x) is type(y):
    print "class x == class y"
else:
    print "class x != class y"

if isinstance(x, list):
    print "x is a instance of list"

