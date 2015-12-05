#!/usr/bin/env python

from multiprocessing.managers import BaseManager

manager = BaseManager(address=("localhost", 5000), authkey="test")
manager.connect()


