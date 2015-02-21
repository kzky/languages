#!/usr/local/ruby
require "rubygems"
require "memcached"

## sample1 for  string
cache = Memcached.new("localhost:11211")
cache.set("user", "kzk", 3)

puts cache.get("user");

sleep 3
begin
  puts cache.get("user")
rescue Memcached::NotFound => e
  p e
end

## sample2 for hash object
hash = {:x => 1, :y => 2, :z => "3"}
cache = Memcached.new("localhost:11211")
cache.set("point", hash)
p cache.get("point")
  

## sample3 for class defined by a user
class User
  @@var = 10
  def initialize(uid, name, age)
    @uid = uid
    @name = name
    @age = age
  end
  
  attr_accessor :uid, :name, :age
end

user = User.new(0, "chisato", 26)
cache = Memcached.new("localhost:11211")
cache.set("user", user)
p cache.get("user")


