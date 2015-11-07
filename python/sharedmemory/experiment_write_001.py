# -*- coding: utf-8 -*-

"""
1. sharedctypes.Array + np.ctypeslib
"""

import numpy as np
import time
import ctypes
import multiprocessing

from multiprocessing import Queue, Process
from multiprocessing import sharedctypes
from numpy import ctypeslib

class Worker(Process):
    """
    Worker process.
    """
    
    def __init__(self, task_queue, result_queue,
                 X_ctypes_in, shape_in,
                 X_ctypes_out, shape_out):
        """
        
        """
        super(Worker, self).__init__()  # this must be necessary.

        self._task_queue = task_queue
        self._result_queue = result_queue
        
        self.X_in = ctypeslib.as_array(X_ctypes_in)
        self.X_in.shape = shape_in

        self.X_out = ctypeslib.as_array(X_ctypes_out.get_obj())
        self.X_out.shape = shape_out

        self.n = shape_in[0]

    def run(self, ):
        """
        
        """
        while True:
            task = self._task_queue.get()  # blocked here
            print "task-{}".format(task[0])
            
            ret = self._compute(task[1], task[2])
            self._result_queue.put(ret)
            
    def _compute(self, s, e):
        st = time.time()
        for i in xrange(s, e):
            for j in xrange(0, self.n):
                self.X_out[i, j] = self.X_in[i, :].dot(self.X_in[j, :])

        et = time.time()

        return et - st
        
def main():

    workers = []
    task_queue = Queue()
    result_queue = Queue()

    # input data
    n = 10000
    m = 100
    X_in = np.random.rand(n, m)
     
    size_in = X_in.size
    shape_in = X_in.shape
    X_in.shape = size_in

    X_ctypes_in = sharedctypes.RawArray(ctypes.c_double, X_in)
    X_in = np.frombuffer(X_ctypes_in, dtype=np.float64, count=size_in)
    X_in.shape = shape_in

    # output data
    X_out = np.random.rand(n, n)
     
    size_out = X_out.size
    shape_out = X_out.shape
    X_out.shape = size_out

    X_ctypes_out = sharedctypes.Array(ctypes.c_double, X_out)
    #X_out = np.frombuffer(X_ctypes_out, dtype=np.float64, count=size_out)
    #X_out.shape = shape_out

    # create worker and start
    concurrency = 4
    unit = n / 4
    for i in xrange(concurrency):
        worker = Worker(task_queue, result_queue,
                        X_ctypes_in, shape_in,
                        X_ctypes_out, shape_out)
        worker.start()
        workers.append(worker)
        
    # put task
    for i in xrange(concurrency):
        task_queue.put((i, unit*i, unit*i+unit))
        pass
        
    # get result of task
    elapsed_times = []
    while True:
        elapsed_time = result_queue.get()
        elapsed_times.append(elapsed_time)

        if len(elapsed_times) == concurrency:
            break

    # stop worker
    for i in xrange(concurrency):
        workers[i].terminate()

    # do math for elapsed time
    print "Elapsed time {} [s]".format(np.max(elapsed_times))


if __name__ == '__main__':
    main()
    

