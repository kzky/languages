# -*- coding: utf-8 -*-

"""
2. globalで定義した変数をワーカーで使う
"""

import numpy as np
import time
import ctypes
from multiprocessing import Queue, Process
from multiprocessing import sharedctypes
from numpy import ctypeslib

class Worker(Process):
    """
    Worker process.
    """
    
    def __init__(self, task_queue, result_queue, X_ctypes, shape):
        """
        
        Arguments:
        - `task_queue`:
        - `result_queue`:
        """
        super(Worker, self).__init__()  # this must be necessary.

        self._task_queue = task_queue
        self._result_queue = result_queue

        # not actually copied as long as passed as sharedctypes
        self.X = ctypeslib.as_array(X_ctypes)
        self.X.shape = shape
        self.n = shape[0]

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
                self.X[i, :].dot(self.X[j, :])

        et = time.time()
        return et - st
        
def main():

    workers = []
    task_queue = Queue()
    result_queue = Queue()

    n = 10000
    m = 100
    X = np.random.rand(n, m)
     
    concurrency = 4
    unit = n / 4
    size = X.size
    shape = X.shape
    X.shape = size

    X_ctypes = sharedctypes.RawArray(ctypes.c_double, X)
    #X = np.frombuffer(X_ctypes, dtype=np.float64, count=size)
    #X.shape = shape

    # create worker and start
    for i in xrange(concurrency):
        worker = Worker(task_queue, result_queue, X_ctypes, shape)
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
    

