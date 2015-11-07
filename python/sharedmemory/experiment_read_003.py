# -*- coding: utf-8 -*-

"""
3. [https://pypi.python.org/pypi/sharedmem/0.3:title=sharedmem]を使う
"""

import numpy as np
import time
import sharedmem
from multiprocessing import Queue, Process

class Worker(Process):
    """
    Worker process.
    """
    
    def __init__(self, task_queue, result_queue, X):
        """
        
        Arguments:
        - `task_queue`:
        - `result_queue`:
        """
        super(Worker, self).__init__()  # this must be necessary.

        self._task_queue = task_queue
        self._result_queue = result_queue
        self.X = X
        self.n = X.shape[0]

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
    X = sharedmem.empty((n, m))
     
    concurrency = 4
    unit = n / 4

    # create worker and start
    for i in xrange(concurrency):
        worker = Worker(task_queue, result_queue, X)
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
    

