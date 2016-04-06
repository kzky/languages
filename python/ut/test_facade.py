'''
Created on 2015/09/15

@author: kzk

Test all test case, this script behave as a facade script to run all tests.

'''
import importlib
import inspect
import os
import unittest
import pprint
from unittest.result import TestResult
from _mysql import result

# Consts
TEST_ROOT_DIR = "sdeepconsole_test"

# Helpers
def fild_all_paths(directory):
    """Return generator whose element is a path to a file.
    """
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)
            
def import_all_modules(directory):
    """Import all modules under directory.
    """
    
    modules = []
    for path in fild_all_paths(directory):
        if path.endswith(".py"):
            
            # replace path format with module format
            mod_name_ = path.replace("/", ".")
            mod_name = mod_name_.replace(".py", "")
            filename = os.path.basename(path)
            
            if not filename.startswith("test_"):
                continue
            
            mod = importlib.import_module(mod_name)
            modules.append(mod)
            
    return modules

def make_suite(directory):
    """Import all modules under directory.
    """
    suite = unittest.TestSuite()
    modules = import_all_modules(directory)

    for module in modules:
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if name.startswith("Test") \
                and not name.endswith("Base") \
                and cls.__module__.startswith(TEST_ROOT_DIR):
                
                suite.addTests(unittest.makeSuite(cls, ))
                
    return suite

def main():
    
    suite = make_suite(TEST_ROOT_DIR)
    unittest.TextTestRunner(verbosity=2).run(suite)
    

if __name__ == '__main__':
    
    main()
    
