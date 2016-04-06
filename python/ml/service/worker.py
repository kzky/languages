#!/usr/bin/env python

# Worker Pool, Worker, and Task
from multiprocessing import Process
from multiprocessing import Queue

import multiprocessing
import logging

class WorkerPool():
    """
    Worker Pool
    """
    NUM_WORKERS = multiprocessing.cpu_count()
    
    def __init__(self, task_queue=None, result_queue=None, num_workers=NUM_WORKERS):
        """
        
        Arguments:
        - `num_worker`: task queue
        - `task_queue`: task queue
        - `result_queue`: result queue
        """

        self._task_queue = task_queue if task_queue is not None else Queue()
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
            self._workers.append(worker)
            worker.start()
            pass
            
        pass

    def terminate(self, ):
        """
        Terminate workers
        """
        for i in xrange(0, self._num_workers):
            self._workers[i].terminate()
            pass

        self._result_queue.close()
        self._task_queue.close()
        pass

class Worker(Process):
    """
    Worker process.
    """

    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)
    logger = logging.getLogger("Worker")

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
            try:
                ret = task.run()
                self._result_queue.put(ret)

            except Exception as e:
                self.logger.info(e)
                self._result_queue.put(None)
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

    def run(self, ):
        """
        """
        result = TaskResult(self.task_name)
        return result

class TaskResult(object):
    """
    """
    
    def __init__(self, val):
        """
        """
        self.val = val
        
        pass
        

class Manager(object):
    """
    """
    
    def __init__(self, ):
        """
        """
        
    def manage(self, ):
        """
        """
        # worker pool
        pool = WorkerPool(num_workers=4)
         
        # put tasks
        print "put tasks"
        num_task = 25
        for i in xrange(0, num_task):
            task = Task(task_name="task %d" % i)
            pool.put(task)
            pass
         
        # get results
        print "get results"
        for i in xrange(0, num_task):
            print pool.get().val
            pass
         
        pool.terminate()

# Explain how-to-use
def main():

    manager = Manager()
    manager.manage()
    
if __name__ == '__main__':
    main()
        
