#!/usr/bin/ruby.1.9.1

require 'rubygems'
require 'scrapi'
require 'open-uri'
require 'pp'

links = Scraper.define do
  process "a[href]", "urls[]"=>"@href"
  result :urls
end

#p links.scrape(URI.parse('http://www.hatena.ne.jp/'), :parser_options => {:char_encoding => 'utf8'}).sort.uniq
p links.scrape(URI.parse('http://www.hatena.ne.jp/')).sort.uniq



