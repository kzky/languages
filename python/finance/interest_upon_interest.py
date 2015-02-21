        

def interest_upon_interest(base, rate, power,):
    """
    
    Arguments:
    - `base`:
    - `rate`:
    - `power`:
    
    """

    return base * (rate ** power)


def main():
    base = 35*10000
    rate = 1.01
    duration = 35
    s = 0
    for i in xrange(duration):
        x_i = interest_upon_interest(base, rate, i)
        print x_i
        s += x_i
        pass
    
    print "total returns: ", s, "[yen]"
    print "total returns: ", s/10000, "[man yen]"

if __name__ == '__main__':
    main()
    pass
