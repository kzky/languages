import sys
import zmq
import pickle as pkl

context = zmq.Context()
sock = context.socket(zmq.SUB)
sock.setsockopt(zmq.SUBSCRIBE, '')

for arg in sys.argv[1:]:
    sock.connect(arg)

while True:
    message = sock.recv()
    data = pkl.loads(message)
    print data

"""
Publish structured data

python pubsub_subscriber_4.py tcp://localhost:8080
"""
