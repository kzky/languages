#!/usr/bin/python

class ClassTest(object):
    """
    Class Test
    """
    
    def __init__(self, a = 10, b = 100):
        """
        method of Class Test
        """
        self.a = a
        self.b = b

        
ct = ClassTest(b = 5, a = 1)
print ct.a
print ct.b
