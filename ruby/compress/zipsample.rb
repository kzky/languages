#!/usr/bin/ruby
# -*- coding: utf-8 -*-

## zipruby version
## (rubyzipもある)
require "rubygems"
require "zipruby"

path = "/home/kzk/downloads/2753-2.zip"
Zip::Archive.open(path) do |archives|
  archives.each do |a|
    unless a.directory?
      # puts a.read
      a.read
    end
  end
end
