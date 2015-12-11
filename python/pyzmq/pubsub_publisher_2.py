import sys
import time
import zmq

context = zmq.Context()
sock = context.socket(zmq.PUB)
sock.bind(sys.argv[1])

while True:
    time.sleep(1)
    sock.send(sys.argv[1] + ':' + time.ctime())

"""
Subscribe multiple publishers

python pubsub_publisher_2.py 'tcp://*:8080'
python pubsub_publisher_2.py 'tcp://*:8081'
"""
