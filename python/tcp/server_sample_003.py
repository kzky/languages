#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import multiprocessing
import socket
import signal
import sys

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print "__exit__ called"
        self._serversocket.shutdown(socket.SHUT_RDWR)
        self._serversocket.close()

    def _parent_main_loop(self):
        while True:
            time.sleep(1)

    def close_socket(self, sig, fem):
        print sig
        self._serversocket.shutdown(socket.SHUT_RDWR)
        self._serversocket.close()

class SockerStreamHandler(object):

    def __init__(self):
        self._conn = None
        self._address = None

    def __call__(self, serversocket):
        while True:
            (self._conn, self._address) = serversocket.accept()
            with self as handler:
                handler.handle()  # NOTE: conn is closed when handle fisnihed.
                
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._conn.shutdown(socket.SHUT_RDWR)
        self._conn.close()

    def handle(self):
        raise NotImplementedError

class EchoHandler(SockerStreamHandler):
    bufsize = 4094
    
    def handle(self):
        print multiprocessing.current_process()
        msg = self._conn.recv(self.bufsize)
        print msg
        self._conn.send(msg)

if __name__ == '__main__':
    argv = sys.argv
    with TCPServer(argv[1], int(argv[2]), 1000, 4) as server:
        #signal.signal(signal.SIGTERM, server.close_socket)
        
        handler = EchoHandler()
        server.start(handler)
