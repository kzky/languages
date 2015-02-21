#!/usr/bin/ruby
# -*- coding: utf-8 -*-

require "rubygems"
require "zipruby"
require "thread"

# url
## http://www.namaraii.com/rubytips/?%E3%82%B9%E3%83%AC%E3%83%83%E3%83%89
## http://techracho.bpsinc.jp/piichan1031/2010_07_05/2030

# unzip function
def unzip(path)
  Zip::Archive.open(path) do |archives|
    archives.each do |a|
      unless a.directory?
        # reading only
        a.read
      end
    end
  end
end

# queue
queue = Queue.new

# push task 
path = "/home/kzk/downloads/*.zip"
Dir.glob(path).each do |p|
  queue.push(p)
end

# create/start therad
num_threads = 3
st = Time.now
(0..(num_threads - 1)).each do |e|
  t = Thread.new() do
    loop do
      p = queue.pop
      unzip(p)
    end
  end
end

# queue empty
while true
  if queue.empty?
    break
  end
end
et = Time.now

puts "total execution time with threading: #{et - st} [s]"

# total execution time with threading: 51.790661849 [s]
