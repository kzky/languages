#!/usr/lib/python

import logging
import sys

from logging import StreamHandler
from logging import Formatter

class LoggingTest(object):
    """
    """
    
    
    FORMAT = '[%(asctime)s::%(levelname)s][%(name)s::%(funcName)s][%(message)s]'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)

    logger = logging.getLogger("LoggingTest")
    
    def __init__(self, ):
        """
        """
        self.a = 100
        self.b = 1000
        
        
    def add(self, a, b):
        """
        
        Arguments:
        - `a`:
        - `b`:
        """
        self.logger.info("add start")
        
        return a + b
    
    
    def __str__(self):
        res = """a: {}\nb: {}
        """.format(self.a, self.b)
        
        return res
        
        
def main():
    c = LoggingTest()
    c.add(5, 5)
    
    print c
    pass

if __name__ == "__main__":
    main()
