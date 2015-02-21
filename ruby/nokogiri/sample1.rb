#!/usr/bin/ruby
# -*- coding: utf-8 -*-

require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'pp'
require 'watir'



# webページの場合 open-uri利用
require 'open-uri'
uri = "https://news.google.co.jp/"
browser = Watir::Browser.new
browser.goto 
doc = Nokogiri::HTML(open(uri))

# css
doc.css("div")
doc.css("#header")
doc.css("div > h1")

# to_html
doc.css("div").each do |elm|
  p elm.to_html
end
   #=> "<div id=\"header\"><h1>title</h1></div>"
   #=> "<div id=\"content\"><h2>content tiltel</h2></div>"
   #=> "<div id=\"footer\">cont end</div>"

# inner_html
doc.css("div").each do |elm|
  p elm.inner_html
end
   #=> "<h1>title</h1>"
   #=> "<h2>content tiltel</h2>"
   #=> "cont end"

# content inner_text
doc.css("div").each do |elm|
  p elm.content
end
   #=> "title"
   #=> "content tiltel"
   #=> "cont end"

# attributes
doc.css("div").each do |elm|
  p elm['id']
end
   #=> "header"
   #=> "content"
   #=> "footer"
