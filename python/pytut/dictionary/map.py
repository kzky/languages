#coding: utf-8

## dictionary はmap型の１つで、hashや連想配列である
## {}とかくからsetに似ている
## 順序がない
## keyの重複は後から定義されたkeyが優先
## キーは、変更できないオブジェクト（数値、文字列、タプル）
## 値は任意のオブジェクト


dict1 = {"key1": 75, "key2": 100,  "key3":1000, "key4":1000, "key3":5}
print "dict = ", dict1

dict2 = {"ja":10, "us":4, "uk":5, "du":1, "dev":[10, 4, 5, 1], "etc":"etc"}
print dict2


dict3 = {(10, 20):"apple", (-10, 3):"bannana"}
print dict3

print dict2
dict2["uk"] = 100
print dict2

print 
print len(dict1), len(dict2), len(dict3)
dict1.update(dict2)
print "update dict1"
print dict1
print dict2

## pop
print "pop"
dict2.pop("ja")
print dict2
val = dict2.pop("in", 10)
print "2nd arg of pop:", val

## popitem
print "\ndelete any element in the dictionary"
while dict2:
    tup = dict2.popitem()
    print tup

## clear
dict3.clear()
print dict3

## in or has_key
print "key" in dict1
print "key1" in dict1 

print

## keys, values (which does not correspond to each other)
print "keys, values"
print dict1.keys()
print dict1.values()

## items
print "items"
dict1.items()
print dict1.items()

