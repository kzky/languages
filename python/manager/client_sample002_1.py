from multiprocessing.managers import BaseManager
class QueueManager(BaseManager):
    pass

QueueManager.register('get_queue')

manager = QueueManager(address=('foo.bar.org', 50000), authkey='abracadabra')
manager.connect()
queue = manager.get_queue()
queue.get()
