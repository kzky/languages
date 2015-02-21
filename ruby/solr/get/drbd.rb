#!/usr/bin/ruby
# -*- coding: utf-8 -*-

require "open-uri"
require "uri"

url = "http://localhost:8983/solr/drbd/select?%s"
param = URI.encode("q=title:ジョジョ OR content:リサリサ&wt=json")

open(url % param).each{ |line|
  print line
}

