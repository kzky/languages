# -*- coding: utf-8 -*-

require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'pp'
require 'kconv'

# webページの場合 open-uri利用
TAG_PATTERN = /^(.*), $/
TITLE_PATTERN = /^(.*) - (\d+) min$/

uri = "http://www.xvideos.com/video3254975/secretary_in_pantyhose_giving_blowjob_fucked_from_behind_by_her_boss_in_the_sitting_roo"
doc = Nokogiri::HTML(open(uri))

exps = doc.css("#main h2")
exps.each do |exp|
  begin
    p exp.content.match(TITLE_PATTERN)[1]
    p exp.content.match(TITLE_PATTERN)[2]
  rescue => e
  end
end

tags = doc.css("#video-tags li")
tags.each do |tag|
  begin
    str = tag.content
    p str.match(TAG_PATTERN)[1]
  rescue => e
  end
end
