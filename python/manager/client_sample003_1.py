"""
can NOT work
"""

from multiprocessing.managers import SyncManager

manager = SyncManager(address=('localhost', 50000), authkey='abracadabra')
manager.register("get_queue")

manager.connect()
queue = manager.get_queue()
print queue
print queue.get()
