#!/usr/bin/ruby
# -*- coding: utf-8 -*-

####################
## address to zipを見つけます
####################

## library
require "net/http"
require "uri"
require "json"

## read file
addresses = []
fin = "/home/kzk/documents/wedding/unknown_zip_adress_tmp.txt"
open(fin).each do |l|
  sl = l.strip().split(",")
  address = [sl[0], sl[1], sl[2], sl[3], sl[4]]
  addresses.push(address)
end

## find zipcode
cnt=0
a="/v1/zipsearch?word="
url = URI.parse("http://api.postalcode.jp")
http_client = Net::HTTP.start(url.host, url.port)
http_client.continue_timeout = 60 * 5
fpout = open("/home/kzk/documents/wedding/found_zip_adress_tmp.txt", "w")
addresses.each do |address|
  cnt+=1
  p cnt
  p URI.escape(address[4].split(" ")[0])
  res = http_client.get(a + URI.escape(address[4].split(" ")[0]))
  begin
    zip = JSON.parse(res.body)["zipcode"]["a1"]["zipcode"]
    fpout.puts "#{address[0]}, #{address[1]}, #{address[2]}, #{zip[0,3]}-#{zip[3,4]}, #{address[4]}"
  rescue => e
    fpout.puts "#{address[0]}, #{address[1]}, #{address[2]}, , #{address[4]}"
  end
end
fpout.close
