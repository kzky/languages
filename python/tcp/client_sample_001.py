#!/usr/bin/env python

from __future__ import print_function
import socket
from contextlib import closing
import cPickle as pkl

def main():
    host = '127.0.0.1'
    port = 4000
    bufsize = 4096
     
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with closing(sock):
        sock.connect((host, port))

        data = {2: 5.6, 4: 6.7}
        sock.send(pkl.dumps(data))  # protobuf should be used.

        msg = sock.recv(bufsize)
        data = pkl.loads(msg)
        print(data)
    return

if __name__ == '__main__':
    main()
