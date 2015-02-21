#!/usr/local/ruby

require "rubygems"
require 'mysql2'

client = Mysql2::Client.new(:host => "localhost", :username => "mysql", :password => "mysql", :database => "book")


val = client.escape("mysql2")
stmt = "INSERT INTO book_info (title, price, author) VALUES ('#{val}',%s, '#{val}')"

(0..10000).each do |price|
  client.query(stmt % price)
end
