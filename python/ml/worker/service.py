#!/usr/bin/env python

# Worker Pool, Worker, and Task
from multiprocessing import Process
from multiprocessing import Queue

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
            #print worker
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
        super(Worker, self).__init__()

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
