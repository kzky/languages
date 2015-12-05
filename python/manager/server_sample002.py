from multiprocessing.managers import BaseManager
import Queue

queue = Queue.Queue()
class QueueManager(BaseManager):
    pass
    
QueueManager.register('get_queue', callable=lambda: queue)

manager = QueueManager(address=('localhost', 50000), authkey='abracadabra')
server = manager.get_server()
server.serve_forever()
