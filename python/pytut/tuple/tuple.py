#coding: utf-8

## 要素の操作はできない、取得のみ
## 操作したい場合はlistに変形してから


t = (1, "a", 10, 1, "b")
u = t[3:]
v = t[4:5]

print "t = ", t
print "u = ", u
print "v = ", v
print len(t)
z = u*3 + v
print "z =", z

print "a" in t
