"""
can NOT work
"""

from multiprocessing.managers import SyncManager

manager = SyncManager(address=('localhost', 50000), authkey='abracadabra')
manager.start()
queue = manager.Queue()
manager.register("get_queue", callable=lambda: queue)
manager.join()

