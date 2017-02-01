#!/usr/bin/env python

import multiprocessing as mp
from multiprocessing import sharedctypes
import time
from numpy import ctypeslib
import os
import numpy as np

class StateArray(object):
    def __init__(self, state, array):
        self.state = state
        self.array = array


class Worker(mp.Process):
    def __init__(self, state_array):
        super(Worker, self).__init__()
        
        self.state_array = state_array

    def run(self, ):
        msg = "Pid={},State={}".format(os.getpid(), self.state_array.state)
        print(msg)


def main():

    buff = np.random.rand(10)
    v0 = 0
    v1 = 1
    sa0 = StateArray(sharedctypes.Value("i", v0), sharedctypes.Array("f", buff))
    sa1 = StateArray(sharedctypes.Value("i", v1), sharedctypes.Array("f", buff))
    #sa0 = StateArray(v0, sharedctypes.Array("f", buff))
    #sa1 = StateArray(v1, sharedctypes.Array("f", buff))
    
    print("State={}, {}".format(v0, v1))

    workers = [
        Worker(sa0).start(),
        Worker(sa1).start(),
        ]

    sa0.state.value = 100
    sa1.state.value = 1

    time.sleep(1)

if __name__ == '__main__':
    main()
