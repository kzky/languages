#!/usr/bin/env python

from multiprocessing.managers import BaseManager

manager = BaseManager(address=("localhost", 50000), authkey="test")
server = manager.get_server()
server.serve_forever()



