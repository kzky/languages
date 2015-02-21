#coding: utf-8

##--- list の色々 ---##

x = [1, -10, 5, 1.4, 6.1, 5, 7, 3]

# len
print len(x)
print "x = ", x

# append, extend, +, *
x.append(-100)
print "append elm:", x

x.append([4, 51])
print "append list:", x

#extend
x.extend([-1, 0])
print "extend:", x

# +, (*で繰り返しも可能)
y = [10, 4, 1]
print "y =", y
print "x + y =", x + y

# del
del y
print "y is deleted\n"

x = ["A", "B", "C", "D"]

print x

## slice (slice によるpopも可能, x[n:m] = [])
print "slice x[1:3] = ['Z', 'Y']"
x[1:3] = ["Z", "Y"] 
print x

## i 番目の要素の変更
print "pop"
#x[1:2] = [] ## or ""
i = x.pop(1)
print i
print x
#x[1:1] = "U"
x.insert(1, "U")
print x

print

# pop
x.pop(1)
print "x.pop(1)", x

# insert
x.insert(1, "U")
print "x.insert(1, 'U')", x

# remove 
print "remove"
x = [1, -10, 5, 1.4, 6.1, 5, 7, 3, 5, 6, 7, 1.4, 1, 1, 10]
print "x =", x
x.remove(5) ## list にない要素を削除するときはエラーがでる
print "x.remove(5)", x

## index, count, in
print "index, count, in"
print x.index(1)
print x.count(1.4)
if 5 in x:
    print "5 in x"

if 100 not in x:
    print "100 not in x"

## sort, reverse
x.reverse()
print x
x.sort()
print x

## range
x = range(0, -7, -2)
print "x =", x


## list to tuple, vice versa
string = "ABCDE"
print "string =", string
print "list(string) =", list(string)

t = ("U", "D", "A")
print t
tl = list(t)
print tl.sort() #Noneになるがsortはされている
print tl

## zip; 複数のリストを結合して、タプルで返す
list1 = ["A", "B"]
list2 = [10, 20, 30, 40]
str = "Hello"

list = zip(list1, list2, str)
print list

## 雑多なlist 
print "\nMisc"

# 文字列の場合
del list
del string
import string

orig = "hello"
li = list(orig) # 文字列 => リスト
#text = "".join(li)  # リスト => 文字列
text = string.join(li, "")  # リスト => 文字列
print li    # ['h', 'e', 'l', 'l', 'o']
print text  # hello
 
# 数値の配列の場合
del str
nums = [1,2,3,4,5]
text = "".join(map(str, nums))  # リスト => 文字列
li   = list(map(int, text))     # 文字列 => リスト
print li    # [1, 2, 3, 4, 5]
print text  # 12345
