import sys
import time
import zmq

context = zmq.Context()
sock = context.socket(zmq.PUSH)
sock.bind(sys.argv[1])

while True:
    time.sleep(1)
    sock.send(sys.argv[1] + ':' + time.ctime())

"""
Boadbalancer

python pushpull_pusher.py "tcp://*:8080"
python pushpull_pusher.py "tcp://*:8081"
"""
