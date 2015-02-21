#coding: utf-8

## make snippet for man's section 3

#coding: utf-8

import re

## file vars
fin_name = "/home/kzk/dataset/man/man3.cmd"
fout_header = "/home/kzk/dataset/man/snippet3/"
ferr = "/home/kzk/dataset/man/err3.cmd"
ferr2 = "/home/kzk/dataset/man/err3.2.cmd"
fpin = file(fin_name, "r")
fperr = file(ferr, "w")
fperr2 = file(ferr2, "w")

## patterns
pat = re.compile("<B>(.*?)</B>")
pat2  = re.compile("<I>(.*?)</I>")
pat3 = re.compile(" <B>(.*?)\(")


for line in fpin: ## outer loop for commands
    func = pat.findall(line)  # function list
    args = pat2.findall(line) # args list
    
    try:
        cmd_name = pat3.search(line).group(1) # command name
    except AttributeError:
        continue
        
    
    snippet = " ".join(func) + ", );" ## snippet without args
    ret_type = str(func[0]) + " "     ## return type

    ## open file written
    try:
        fpout = file(fout_header + cmd_name, "w")
    except IOError:
        fperr.write(line);
        continue
        
    
    ## paste func and args, and arrange detail to make snippet
    count = 0
    i = 0
    flag = True
    while True: 
	
        if snippet[i:i+1] == ",":
            try:
                repl = args[count] + ", "
            except IndexError:
                flag = False
                fperr2.write(line)

            snippet = snippet[0:i] + snippet[i:].replace(", ", repl, 1)
            count += 1
            i+=len(repl) 
            
        i += 1;
        if snippet[i:i+1] == ";":
            snippet = re.sub(";", "\n$000", snippet, 1)
            break

    if flag == False:
        continue

    ## details to make snippet
    snippet = snippet.replace("}", "}, ")
    snippet = snippet[0:len(snippet)-2]
    snippet = snippet.replace(", )", ");")
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

        snippet = snippet.replace("   ", "")
        snippet = snippet.replace("  ", "")
        snippet = snippet.replace(":  ", ":")
        snippet = snippet.replace(": ", ":")
        snippet = snippet.replace(",$", ", $")
    
        
    fpout.write("# name: " + cmd_name + "\n")
    fpout.write("# key: " + cmd_name + "\n")
    fpout.write("# --" + "\n")
    fpout.write(snippet)
    
    ## close file written
    fpout.close()

fpin.close()
fperr.close()
fperr2.close()
