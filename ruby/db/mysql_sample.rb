#!/usr/local/ruby

require "rubygems"
require 'mysql2'

client = Mysql2::Client.new(:host => "localhost", :username => "mysql", :password => "mysql", :database => "book")
client.query("select title, price, author from book_info").each do |dat|
  #puts(dat["price"])
  p dat
end

=begin
val1 = 123
val2 = client.escape('abc')
client.query("INSERT INTO tblname (col1,col2) VALUES (#{val1},'#{val2}')")
=end
