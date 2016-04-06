#!/usr/bin/env python

'''
while loop vs for loop in speed
'''

import time

def task():
    return 10.0 * 10.0

def forloop(maxitr):
    for i in xrange(maxitr):
        task()
        pass
    pass

def whileloop(maxitr):

    i = 0
    while i < maxitr:
        task()
        i += 1
    pass

def timer(f, **kwargs):
    st = time.time()
    f(**kwargs)
    et = time.time()
    print "{}: elapsed time {}".format(f.func_name, et - st)
    
    pass


def main():
    kwargs = {
        "maxitr": 100000000
    }
    timer(whileloop, **kwargs)
    timer(forloop, **kwargs)


if __name__ == '__main__':
    main()
