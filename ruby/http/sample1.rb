#! /usr/bin/ruby1.9.3

require "net/http"
require "uri"
require "rubygems"
require "pp"
require "kconv"

uri = URI.parse("http://eow.alc.co.jp/search?q=revocation")

res = Net::HTTP.start(uri.host, uri.port) { |http|
  http.get("/search?q=revocation")
}

pp res.body.toutf8




