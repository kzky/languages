#!/usr/bin/ruby

def method (str)
  hoge = "hoge"
  if str != ""
    hoge.concat(str)
  end
  return hoge
end

str = "foo"
p method(str)
