import sys
import time
import zmq
import pickle as pkl

context = zmq.Context()
sock = context.socket(zmq.PUB)
sock.bind(sys.argv[1])

while True:
    time.sleep(1)
    data = {
        "host": sys.argv[1],
        "time": time.ctime()
    }
    sock.send(pkl.dumps(data))

"""
Publish structured data

python pubsub_publisher_4.py 'tcp://*:8080'
"""
