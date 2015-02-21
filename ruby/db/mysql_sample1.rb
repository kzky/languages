#!/usr/local/ruby

require "rubygems"
require 'mysql'


client= Mysql.connect('localhost', 'mysql', 'mysql', 'book')

stmt = client.prepare('INSERT INTO book_info (title, price, author) VALUES (?,?,?)')

(0...10000).each do |price|
  stmt.execute("mysql", price, "mysql1")
end



stmt.execute "abc", 123, "abc"
