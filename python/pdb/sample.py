#!/usr/bin/env python

import numpy as np

x = np.random.randint(10)
y = np.random.randint(10)
z = np.random.randint(10)

def main():

    if x > 5:
        print x

    if y < 10:
        print y

    if z == 5:
        print z
    
    pass

if __name__ == '__main__':
    main()
