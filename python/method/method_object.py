#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Methodが変数に渡せることを確かめる
"""

class ClassName(object):
    """
    """
        
    def __init__(self, ):
        """
        """
        self.var = self.add

        pass

    def add(self, a, b):
        """
        
        Arguments:
        - `a`:
        - `b`:
        """
        return a + b

def main():

    a = ClassName()
    print type(a.var)
    print a.var(1, 2)
    pass


if __name__ == '__main__':
    main()    

