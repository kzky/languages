#coding: utf-8

import re

fin_name = "/home/kzk/language/python/parse/man/fread.cmd"
fpin = file(fin_name)

pat = re.compile("<B>(.*?)</B>")
pat2  = re.compile("<I>(.*?)</I>")
pat3 = re.compile(" <B>(.*?)\(")

for line in fpin:
    func = pat.findall(line)
    args = pat2.findall(line)
    cmd_name = pat3.search(line).group(1)

snippet = " ".join(func) + ", );"

ret_type = str(func[0]) + " "

print snippet

## paste func and args, and arrange details for yasnippet
count = 0
i = 0
while True:

    if snippet[i:i+1] == ",":
        #repl = "${" + str(count+2) + ":"+ args[count] + "}"
        repl = args[count] + ", "
        snippet = snippet[0:i] + snippet[i:].replace(",", repl, 1)
        count += 1
        i+=len(repl)
        
    i += 1;
    if snippet[i:i+1] == ";":
        snippet = snippet.replace(";", "\n$000", 1)
        break

snippet = snippet.replace("}", "}, ")
snippet = snippet[0:len(snippet)-2]
snippet = snippet.replace(",  )", ");")
snippet = snippet.replace("* ", "*")
snippet = snippet.replace(ret_type, "${1:" + ret_type +"}", 1)

i = 0
count = 2
while True:
    
    if snippet[i:i+1] == "(":
        snippet = snippet[0:i] + snippet[i:].replace("(", "(${" + str(count) + ":", 1)
        i += 3 + len(args[count - 2])
        count += 1
        print snippet
        
    if snippet[i:i+1] == ",":
        snippet = snippet[0:i] + snippet[i:].replace(",", "},${" + str(count) + ":", 1)
        i += 5 
        count += 1
        print snippet
        
    if snippet[i:i+1] == ")":
        snippet = snippet.replace(")", "})")
        break

    i += 1

snippet = snippet.replace("  ", "")
snippet = snippet.replace(",$", ", $")

print snippet 
print
print "# name: " + cmd_name
print "# key: " + cmd_name
print "# --"
print snippet

