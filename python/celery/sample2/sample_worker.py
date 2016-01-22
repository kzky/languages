from celery import Celery
import time
import celeryconfig
from multiprocessing import Process, Queue
import random
from flask_sqlalchemy import SQLAlchemy

app = Celery("tasks")
app.config_from_object(celeryconfig)
db = SQLAlchemy(app)

class Worker(Process):

    def __init__(self, in_queue, out_queue):
        super(Worker, self).__init__()

        self.in_queue = in_queue
        self.out_queue = out_queue
        
    def run(self, ):
        while True:
            task = self.in_queue.get()
            res = self.do_task(task)
            self.out_queue.put(res)

    def do_task(self, task):
        return task * 10
    
class WorkerPool(object):
    
    def __init__(self, num=4):
        self.num = num
        self.pool = []
        self.in_queue = Queue()
        self.out_queue = Queue()
        
        for i in xrange(num):
            p = Worker(self.in_queue, self.out_queue)
            p.start()
            self.pool.append(p)

    def stop(self, ):
        [p.stop() for p in self.pool]
            
    def put(self, task):
        self.in_queue.put(task)

    def get(self, ):
        return self.out_queue.get()

@app.task
def add(task):
    pool = WorkerPool()

    for i in xrange(100):
        pool.put(random.random())

    for i in xrange(100):
        print pool.get()
        
    return "Finished!"
