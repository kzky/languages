#!/usr/bin/env python

"""
Multiprocess sample with function and pre-defined pool.

http://docs.python.jp/2.7/library/multiprocessing.html
"""

from multiprocessing import Pool

import multiprocessing
import copy_reg
from types import MethodType

"""
"cPickle.PicklingError: Can't pickle <type 'instancemethod'>: attribute lookup __builtin__.instancemethod failed" will be solved by copy_reg.
See http://docs.python.jp/2/library/copy_reg.html
"""
#class Worker(object):
#    """
#    """
#    
#    def __init__(self, a):
#        """
#        """
#        self.a = a
# 
#        copy_reg.pickle(MethodType, self.func)
# 
#        
#    def func(self, x):
#        """
#        
#        Arguments:
#        - `x`:
#        """
#        return self.a * x


class Math(object):
    """
    """
    
    def __init__(self, ):
        """
        """
        
    def pow(self, x):
        """
        """
        return x * x
        

# have to be defined in global scope
def pool_helper(x):
    s = Math()
    return s.pow(x)

def callback_func(x):  # task one argument
    print "callbacked!"
    print x  # result is stored

def main():
    NUM_WORKERS = multiprocessing.cpu_count()
    pool = Pool(processes=NUM_WORKERS)

    # have to be defined in global scope
    result = pool.apply_async(func=pool_helper, args=(10, ), callback=callback_func)
    print result.get(timeout=1)
        
if __name__ == '__main__':
    main()


