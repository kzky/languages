#coding: utf-8

for i in range(0, 10):
    print i
else:
    print "END"

print


l = [15, 4, 5, 6, 10]
print l
for i in l:
    print i

print


s = {4, 4, 5, 10}
print s
for i in s:
    print i

dictionary = {"apple":10, "bannna": 50, "orarnge":-10}
print dictionary
for i in dictionary:
    print i, ":\t", dictionary[i]

for k, v in dictionary.iteritems():
    print k, ":\t", v


print 
tup = (10, 10, 5, -10, "a")
print tup
for i in tup:
    print i

print "\nenumerate"

for i, c in enumerate("abc"):
    print i, c

print max(["1", "3", "-1", "100"]) ## 数字の文字列ではダメ！
