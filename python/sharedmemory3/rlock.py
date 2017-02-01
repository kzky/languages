#!/usr/bin/env python

import multiprocessing as mp
from multiprocessing import sharedctypes
import time
from numpy import ctypeslib
import os
import numpy as np

class StateValue(object):
    def __init__(self, state, ):
        self.state = state


class Worker(mp.Process):
    def __init__(self, state_value):
        super(Worker, self).__init__()
        
        self.state_value = state_value

    def run(self, ):
        with self.state_value.state.get_lock():
            msg = "Pid={},State={}".format(os.getpid(), self.state_value.state)
        print(msg)


def main():

    buff = np.random.rand(10)
    v0 = 10
    sa0 = StateValue(sharedctypes.Value("i", v0))
    
    sa0.state.acquire()

    workers = [
        Worker(sa0).start(),
        ]

    time.sleep(5)
    sa0.state.release()
    time.sleep(5)

if __name__ == '__main__':
    main()
