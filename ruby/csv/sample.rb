#! /usr/bin/ruby
# -*- coding: utf-8 -*-

require "csv";
count = 0

## あまり速くない
CSV.open("/home/kzk/dataset/drbd/73789/contents_1000.tsv", mode="r", fs="\t") do |line|
  print(count, ": \n")
  #puts(line)
  count += 1
end
