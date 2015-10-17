#!/usr/bin/env python

# yield sample.
# yield is used as a producer and for-statement is used as a comsumer.
# function/method usesing yield in those has a state like a class


def tissue_box():
    for i in xrange(5):
        print "yield %d !" % i
        yield i

def main():
    for i in tissue_box():
        print "consume %d !" % i

if __name__ == '__main__':
    main()



    
