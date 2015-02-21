#!/usr/bin/ruby

def hoge(*params) 
  
  for i in params:
      puts(i)
  end
  
end

hoge(1, 2, 3, 4)
