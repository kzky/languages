#!/usr/lib/python


var = "global var";

def write_stdout():
    # var = "local var" error
    
    global var
    print var

write_stdout()
