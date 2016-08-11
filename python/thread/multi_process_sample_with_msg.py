#!/usr/bin/env python

"""
Multiprocess sample, creating WorkerPool by oneself

http://docs.python.jp/2.7/library/multiprocessing.html
"""

from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import JoinableQueue

import multiprocessing

class M(object):
    DO_TASK = "do"
    STOP = "stop"

class WorkerPool():
    """
    Worker Pool
    """
    NUM_WORKERS = multiprocessing.cpu_count()
    
    def __init__(self, msg_queue=None, result_queue=None, num_workers=NUM_WORKERS):
        """
        
        Arguments:
        - `num_worker`: int
        - `msg_queue`: msg queue
        - `result_queue`: result queue
        """

        self._msg_queue = msg_queue if msg_queue is not None else Queue()
        self._result_queue = result_queue if result_queue is not None else Queue()

        self._num_workers = num_workers
        self._workers = []
        
        self._start_workers()
        
        pass

    def put(self, task):
        """
        Put task to worker
        Arguments:
        - `task`:
        """

        self._msg_queue.put(task)
        
        pass

    def get(self, ):
        """
        Get result from queue
        """

        return self._result_queue.get()

    def _start_workers(self, ):
        """
        Append self.num_workersworker to _workers
        """

        for i in xrange(0, self._num_workers):
            worker = Worker(self._msg_queue, self._result_queue)
            self._workers.append(worker)
            worker.start()
            pass
            
        pass

    def stop(self, ):
        for i in xrange(0, self._num_workers):
            self._msg_queue.put(M.STOP)
            
    def terminate(self, ):
        """
        Terminate workers
        """
        for i in xrange(0, self._num_workers):
            self._workers[i].terminate()

        self._result_queue.close()
        self._msg_queue.close()

class Worker(Process):
    """
    Worker process.
    """
    
    def __init__(self, msg_queue, result_queue):
        """
        
        Arguments:
        - `msg_queue`:
        - `result_queue`:
        """
        super(Worker, self).__init__()  # this must be necessary.

        self._msg_queue = msg_queue
        self._result_queue = result_queue
        self.daemon = True

    def run(self, ):
        """
        
        """
        
        while True:
            msg = self._msg_queue.get()  # blocked here
            if msg == M.DO_TASK:
                print "Do a task"
                self._result_queue.put("Task result")
            if msg == M.STOP:
                print "Stop worker"
                break
        
def main():
    # worker pool
    pool = WorkerPool(num_workers=4)

    # put tasks
    print "put tasks"
    num_task = 100
    for i in xrange(0, num_task):
        pool.put(M.DO_TASK)
        pass

    # get results
    print "get results"
    for i in xrange(0, num_task):
        print pool.get()

    pool.stop()

if __name__ == '__main__':
    main()
