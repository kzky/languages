# -*- coding: utf-8 -*-

require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'pp'

# webページの場合 open-uri利用
require 'open-uri'
uri = "http://www.xvideos.com/tags/japanese/3/s:uploaddate/m:10min_more#!/usr/bin/ruby"
doc = Nokogiri::HTML(open(uri))

pp doc

