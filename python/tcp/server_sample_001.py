#!/usr/bin/env python

from __future__ import print_function
import socket
from contextlib import closing
import cPickle as pkl

def main():
    host = '127.0.0.1'
    port = 4000
    backlog = 10
    bufsize = 4096
     
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with closing(sock):
        sock.bind((host, port))
        sock.listen(backlog)
        while True:
            conn, address = sock.accept()
            print(address)
            with closing(conn):
                msg = conn.recv(bufsize)
                data = pkl.loads(msg)
                print(data)
                conn.send(pkl.dumps(data))
    return

if __name__ == '__main__':
    main()
