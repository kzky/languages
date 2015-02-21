
## only hash args
def foo(options = {})

  options.each do |key, value|
    print(key, ": ", value, "\n")
  end

end

## arg, hash args
def fuga(condition, options = {})

  options.each do |key, value|
    print(key, ": ", value, "\n")
  end

end

## only key
def goo(key)
  hash = {:a => 10}
  p hash[key]
end

foo(:a => "hoge", :b => "fuga", :c => "foo", :d => "hoo")
fuga(:a => "hoge", :b => "fuga", :c => "foo", :d => "hoo")
fuga(10, :a => "hoge", :b => "fuga", :c => "foo", :d => "hoo")
goo(:a)
