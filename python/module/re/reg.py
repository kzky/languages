#condig: utf-8

import re

s = ""
fp = file("/home/kzk/tex/NLP/mini_report_2.tex", "r")
pat = re.compile("\\.*\{.*\}") ## find latex command
n = 1

## from file
print "From file directly"
for line in fp:
    s += line
    x = pat.search(line)
    if x is not None:
        print "%d, %s" % (n, line)
        n += 1

## from very long string containing "\n"
print "From very long string containing '\n'"
lines = s.split("\n")
pat = re.compile("\\.*\{(.*)\}")
n = 1
for i in range(0, len(lines)):
    x = pat.search(lines[i])
    if x is not None:
        print "(", n, ")", x.group(1)
        n += 1


## findall
print "findall fucntion from string"
find = pat.findall(s)
print find

fp.close()        
