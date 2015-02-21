#!/usr/bin/ruby
# -*- coding: utf-8 -*-

class Test
  attr_accessor :class_var
  @@class_var = ""

  def getClassVar
    if @@class_var == ""
      @@class_var = "class_var"
      return @@class_var
    else
      return @@class_var
    end
  end

  def setClassVar (class_var)
    @@class_var = class_var
  end
end

## instance_1
test1 = Test.new()
p test1.getClassVar

## instance_2
test2 = Test.new()
p test1.getClassVar

## instance_2のクラス変数に100をセット
test2.setClassVar(100)

## instance_1のクラス変数でアクセス．
p test1.getClassVar


