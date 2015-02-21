#!/usr/bin/ruby
# -*- coding: utf-8 -*-

require "rubygems"
require "zipruby"

# url
## http://www.namaraii.com/rubytips/?%E3%82%B9%E3%83%AC%E3%83%83%E3%83%89
## http://techracho.bpsinc.jp/piichan1031/2010_07_05/2030
## 

##############
## zip fileを解凍する
##############

## unzip function
def unzip(path)
  Zip::Archive.open(path) do |archives|
    archives.each do |a|
      unless a.directory?
        ## reading only
        a.read
      end
    end
  end
end

## non-thread for comparison
path = "/home/kzk/downloads/*.zip"
st = Time.now
for fpath in Dir.glob(path)
  unzip(fpath)
end
et = Time.now
puts "total execution time without threading: #{et - st} [s]"

# total execution time with threading: 54.494340185 [s]
