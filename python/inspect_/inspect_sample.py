'''
Created on 2015/09/15

@author: kzk
'''

import fnmatch
import glob
from inspect import isclass
import inspect
import os

from pkg1.pkg11 import mod11
import pkgutil
import importlib

if __name__ == '__main__':

    print "# Inspect classes in a module if its module name (xxx.py) is known."  
    for name, obj in inspect.getmembers(mod11, isclass):
#         if inspect.isclass(obj):
#             print name
#             print type(obj), obj
#             obj_ = obj()
#             print type(obj_)
#             print obj_
  
        print name
        print type(obj), obj
        obj_ = obj()
        print type(obj_)
        print obj_    
    
    print ""
    print ""
    
    
    print "# Walk"
    for root, dirnames, filenames in os.walk("pkg1"):
        print "## outer loop"
        print root, dirnames, filenames
        for filename in fnmatch.filter(filenames, "*.py"):
            print "## inner loop"
            print dirnames
            print filename
    

    print ""
    print ""
    
    print "Qiita version scan all files"
    
    def fild_all_files(directory):
        for root, dirs, files in os.walk(directory):
            yield root
            for file in files:
                yield os.path.join(root, file)


    for file in fild_all_files("pkg1"):
        print file
        
        
    print ""
    print ""
     
    print "Load module by string and Get class"
     
    def fild_all_paths(directory):
        for root, dirs, files in os.walk(directory):
            yield root
            for file in files:
                yield os.path.join(root, file)
 
 
    modules = []
    for path in fild_all_paths("pkg1"):
        if path.endswith(".py"):
            mod_name_ = path.replace("/", ".")
            mod_name = mod_name_.replace(".py", "")
            if mod_name.endswith("__"):
                continue
              
            print mod_name
            mod = importlib.import_module(mod_name)
            modules.append(mod)
            
    print modules
          
    for module in modules:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            print name, obj


#     modulenames = ["pkg1.pkg11.mod11", "pkg1.pkg12.mod12", "pkg1.pkg12.pkg121.mod121"]
#     modules = []
#     for modulename in modulenames:
#         module = importlib.import_module(modulename)
#         modules.append(module)
#     print modules
  
  
#     print ""
#     print ""
#     
#     
#     print "Pgkutil"
#     
#     for loader, name, ispkg in pkgutil.walk_packages(path=["pkg1"]):
#         print loader, name, ispkg
    
    
    
    
