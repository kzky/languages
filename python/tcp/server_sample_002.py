#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import multiprocessing
import socket


class TCPServer(object):

    def __init__(self, host, port, qsize, processes):
        self._serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._serversocket.bind((host, port))
        self._serversocket.listen(qsize)
        self.processes = processes

    def start(self, handler):
        for i in range(self.processes):
            p = multiprocessing.Process(target=handler,
                                        args=(self._serversocket, ))
            p.daemon = True
            p.start()

        self._parent_main_loop()

    def _parent_main_loop(self):
        while True:
            time.sleep(1)

class SockerStreamHandler(object):

    def __init__(self):
        self._conn = None
        self._address = None

    def __call__(self, serversocket):
        while True:
            (self._conn, self._address) = serversocket.accept()
            with self:
                self.handle()  # NOTE: conn is closed when handle fisnihed.
                
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self._conn.shutdown(socket.SHUT_RDWR)
        self._conn.close()

    def handle(self):
        raise NotImplementedError

class HelloWorldHendler(SockerStreamHandler):
    def handle(self):
        self._conn.send('Hello, World!\n')


if __name__ == '__main__':
    server = TCPServer("localhost", 50000, 1000, 4)
    handler = HelloWorldHendler()
    server.start(handler)
