#!/usr/bin/env python

from multiprocessing.managers import BaseManager

class QueueManager(BaseManager):
    pass

QueueManager.register('get_queue')

manager = QueueManager(address=('localhost', 50000), authkey='abracadabra')
manager.connect()
queue = manager.get_queue()
queue.put('hello')
