#
#   Request-reply service in Python
#   Connects REP socket to tcp://localhost:5560
#   Expects "Hello" from client, replies with "World"
#
import zmq
import sys

context = zmq.Context()
socket = context.socket(zmq.REP)
host_port = sys.argv[1]
socket.connect(host_port)

while True:
    message = socket.recv()
    print("Received request: %s" % message)
    socket.send(b"World")

"""
python queue_server_2.py "tcp://localhost:5560"
python queue_server_2.py "tcp://localhost:5561"
"""
