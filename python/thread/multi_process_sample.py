#!/usr/bin/env python

"""
Multiprocess sample, creating WorkerPool by oneself

http://docs.python.jp/2.7/library/multiprocessing.html
"""

from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import JoinableQueue

import multiprocessing

class WorkerPool():
    """
    Worker Pool
    """
    NUM_WORKERS = multiprocessing.cpu_count()
    
    def __init__(self, num_workers=NUM_WORKERS, task_queue=Queue(), result_queue=Queue()):
        """
        
        Arguments:
        - `num_worker`: task queue
        - `task_queue`: task queue
        - `result_queue`: result queue
        """

        self._task_queue = task_queue
        self._result_queue = result_queue

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

        self._task_queue.put(task)
        
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
            worker = Worker(self._task_queue, self._result_queue)
            worker.start()
            self._workers.append(worker)
            pass
            
        pass

    def terminate(self, ):
        """
        Terminate workers
        """
        for i in xrange(0, self._num_workers):
            self._workers[i].terminate()
            pass

        pass
        pass


class Worker(Process):
    """
    Worker process.
    """
    
    def __init__(self, task_queue, result_queue):
        """
        
        Arguments:
        - `task_queue`:
        - `result_queue`:
        """
        super(Worker, self).__init__()  # this must be necessary.

        self._task_queue = task_queue
        self._result_queue = result_queue
        
        pass

    def run(self, ):
        """
        
        """

        while True:
            task = self._task_queue.get()  # blocked here
            ret = task.do_task()
            self._result_queue.put(ret)
            
            pass
            
        pass
        pass


class Task(object):
    """
    Task
    """
    
    def __init__(self, task_name=""):
        """
        """
        self.task_name = task_name
        pass

    def do_task(self, ):
        """
        """
        
        return self.task_name + " is finished."

        
def main():
    # worker pool
    pool = WorkerPool(num_workers=4)

    # put tasks
    print "put tasks"
    num_task = 25
    for i in xrange(0, num_task):
        pool.put(Task(task_name="task %d" % i))
        pass

    # get results
    print "get results"
    for i in xrange(0, num_task):
        print pool.get()
        pass

    pool.terminate()

if __name__ == '__main__':
    main()
