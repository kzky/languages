#!/usr/bin/env python

class Utility(object):
    """
    """
    
    def __init__(self, ):
        """
        """

    @staticmethod
    def get_class(kls):
        parts = kls.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m
        