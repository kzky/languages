#coding: utf-8

## __init__(self, ...): コンストラクタ

class test:
    
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __init__(self, a, b, c):
        self.a = a * 2
        self.b = b * 2
        self.c = c * 2
    
    def add(self):
        return self.a + self.b
        
    def sub(self):
        return self.a - self.b

    def mul(self, c):
        self.a *= c
        self.b *= c

a = 10
b = 5
c = 1
t = test(a, b, c)
print t.a, t.b

t.mul(10)
print t.a, t.b



