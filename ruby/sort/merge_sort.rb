# -*- coding: utf-8 -*-
def mergesort list
  return _mergesort_ list.dup  # 副作用で配列が壊れるので、複製を渡す
end

$cnt_mergesort = 0
def _mergesort_ list
  if (len = list.size) <= 1 then
    return list
  end
  
  # pop メソッドの返す値と副作用の両方を利用して、list を二分する
  list2 = list.pop(len >> 1) ## 1bit right-shift
  puts "in _mergesort_:: list: #{list}, list2: #{list2}"

  # _merge_も再帰的に呼ばれることになる．
  return _merge_(_mergesort_(list), _mergesort_(list2)) 
end

def _merge_ list1, list2
  len1, len2 = list1.size, list2.size
  puts "in _merge_:: list1: #{list1}, list2: #{list2}"
  result = Array.new(len1 + len2)
  a, b = list1[0], list2[0]
  i, j, k = 0, 0, 0
  loop {
    if a <= b then
      result[i] = a
      i += 1 ; j += 1
      break unless j < len1
      a = list1[j]
    else
      result[i] = b
      i += 1 ; k += 1
      break unless k < len2
      b = list2[k]
    end
  }
  while j < len1 do
    result[i] = list1[j]
    i += 1 ; j += 1
  end
  while k < len2 do
    result[i] = list2[k]
    i += 1 ; k += 1
  end
  return result
end

## main
if __FILE__ == $0
  list = [4,5,6,7,8,9,0,1,2,3].shuffle
  puts "list: #{list}"

  slist = mergesort  list
  p slist
end
