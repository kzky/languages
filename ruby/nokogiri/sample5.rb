# -*- coding: utf-8 -*-

require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'pp'
require 'kconv'

# webページの場合 open-uri利用
TAG_PATTERN = /^(.*), $/
TITLE_PATTERN = /^(.*) - (\d+) min$/
SELECTOR = "div ul li:last-child .sel"
#SELECTOR = "div ul li .sel"

uri = "http://www.xvideos.com/tags/japanese/10000/s:uploaddate/m:allduration/d:all"
doc = Nokogiri::HTML(open(uri))
p doc.css(SELECTOR)

doc.css(SELECTOR).each do |elm|
  p elm.content
end
