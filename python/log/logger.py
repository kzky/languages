#!/usr/lib/python

import logging
import sys

from logging import StreamHandler
from logging import Formatter

class LoggingTest(object):
    """
    """
    
    
    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)

    logger = logging.getLogger("LoggingTest")
    
    def __init__(self, ):
        """
        """
        
    def add(self, a, b):
        """
        
        Arguments:
        - `a`:
        - `b`:
        """
        self.logger.info("add start")
        
        return a + b
        
def main():
    c = LoggingTest()
    c.add(5, 5)
    pass

if __name__ == "__main__":
    main()
