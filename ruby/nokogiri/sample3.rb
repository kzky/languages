# -*- coding: utf-8 -*-

require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'pp'
require 'kconv'
#require 'watir' # webbrower opend
require 'watir-webdriver'
require 'headless'

# webページの場合 open-uri利用
require 'open-uri'
#uri = "http://www.xvideos.com/tags/japanese/2/s:uploaddate/m:3-10min/d:week"
uri = "http://www.xvideos.com/tags/japanese/7/s:rating/m:3-10min/d:week"

headless = Headless.new
headless.start
browser = Watir::Browser.new
browser.goto(uri)
doc = Nokogiri::HTML.parse(browser.html)

contents = doc.css(".thumbBlock")

contents.each do |content|
  p content['id'].split("_")[1]
end
puts()

contents.css("img").each do |img|
  p img['src']
end
puts()

contents.css(".thumb a").each do |url|
  p url["href"]
end
puts()

#contents.css("script").each do |script|
#  p script.inner_text
#end

browser.close()
headless.destroy
