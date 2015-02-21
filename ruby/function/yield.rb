def test7
  for i in [1,2,3]
   yield i, "hoge"
  end
end

test7{|x,y| print  x, y, "\n"}


puts("");

def test8(val)
  for i in [1,2,3,4,5]
   yield i, "hoge", val  if i > val
  end
end

test8(0){|x, y, z| print x, y, z, "\n"}
test8(2){|x, y, z| print x, y, z, "\n"}
