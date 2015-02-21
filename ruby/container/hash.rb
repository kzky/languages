#! /usr/bin/ruby


hash = {:a => "goo", :b => "foo", :c => "hoge"}
puts(hash.each_key)
puts(hash.each_value)

def key_test(key)
  puts(hash[:key])
end

for key in hash.each_key do
  puts(hash[key])
end

print("a is",  hash[:a], "\n")

hash.each do |key, value|
  print key, ": ", value, "\n"
end

