from celery import Celery
import celeryconfig
from multiprocessing import Process, Queue
import random
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String
from sqlalchemy import Sequence
from sqlalchemy.ext.declarative import declarative_base
import gevent


# Varialbes
app = Celery("tasks")
app.config_from_object(celeryconfig)
engine = create_engine('mysql://root:root@localhost/celery_sample_db', echo=True)
Session = sessionmaker(bind=engine)
session = Session()

# OR
Base = declarative_base()
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    name = Column(String(50))
    email = Column(String(50))
    password = Column(String(12))

    def __repr__(self):
        return "<User(name='%s', email='%s', password='%s')>" % (
            self.name, self.email, self.password)

# Create tables
Base.metadata.create_all(engine)

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

    def do_task(self, i):
        user = User(name='user_{}'.format(i),
                    email='user@example.com',
                    password="password")

        try:
            session.add(user)
            session.commit()

        except gevent.Timeout as e:
            #session.invalidate()
            session.close()
            #print e
        except Exception as e:
            session.rollback()
            #print e

        return user.name
    
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

    def shutdown(self, ):
        [p.terminate() for p in self.pool]
            
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
        #print i

    pool.shutdown()
    print "pool shutdowned"

    return "Finished!"
