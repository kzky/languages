#!/usr/bin/python
# -*- coding: utf-8 -*-

## 文字列中で式展開をする

# 文字列 in 文字列
instr = "test"
print "%(instr)s" % locals()

# 数値 in 文字列
num = 10
print "%(num)d" % locals()


