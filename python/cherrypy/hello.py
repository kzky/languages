#!/usr/bin/env python

import random
import string

import cherrypy
import json


class SharedData(object):
    """
    """
    data = {}
    data[10] = 10
    data[100] = 100
    data[1000] = 1000
    
    def __init__(self, ):
        """
        """
                
        pass

    
class StringGenerator(object):
    
    @cherrypy.expose
    @cherrypy.tools.json_out()  # return val as json
    def index(self):
        return SharedData.data
        #return str(SharedData.data)
        #return "Hello world!"

    @cherrypy.expose
    def generate(self, length=8):
        return ''.join(random.sample(string.hexdigits, int(length)))

if __name__ == '__main__':
    cherrypy.quickstart(StringGenerator())
