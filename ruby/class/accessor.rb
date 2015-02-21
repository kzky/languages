#!/usr/bin/ruby
# -*- coding: utf-8 -*-

class LifeForm
  attr_reader :name

  def initialize(name)
    @name = name
  end
  
  def get_name()
    return @name
  end

  def print_name()
    puts(get_name)
  end

  protected :get_name ## methodのアクセッサーはメソッド定義より後ろに書く．
end

class Person < LifeForm
end

## parentClassのインスタンスを作成
life_form = LifeForm.new("test")

## subClassのインスタンスを作成
person = Person.new("hoge")

## @nameをプリント
life_form.print_name()
person.print_name() ## インスタンス変数/インスタンスメソッドを引き継いでいる

 
