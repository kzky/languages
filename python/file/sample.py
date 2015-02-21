#!/usr/bin/python

fin = "/home/kzk/.zshenv"
fpin = open(fin, "r")

# redline
print "### readline ###"
print fpin.readline()

# seek
fpin.seek(0)

# readlines
print "### readlines ###"
print fpin.readlines()
fpin.seek(0)

# for read
print "### readline with for-loop ###"
for line in fpin:
    print line,
    
fpin.seek(0)

# for read
print 
print "### readline with for-loop and strip ###"
for line in fpin:
    print line.strip()

    



