#!/usr/bin/ruby
# -*- coding: utf-8 -*-

require 'MeCab'
file_in = "/home/kzk/project/website/ero/orgasmo/data/actresses.txt"

begin
  
  print MeCab::VERSION, "\n"    
  model = MeCab::Model.new(ARGV.join(" "))
  tagger = model.createTagger()

  open(file_in).each do |line|
    #n = tagger.parseToNode(line)
    
    parsed_sentence = tagger.parse(line)
    puts(parsed_sentence)
    
    #while n do
      #print n.surface,  "\t", n.feature, "\t", n.cost, "\n"
      #n = n.next
    #end
    #print "EOS\n";
    
  end


end
