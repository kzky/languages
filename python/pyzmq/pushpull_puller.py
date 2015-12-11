import sys
import zmq

context = zmq.Context()
sock = context.socket(zmq.PULL)

for arg in sys.argv[1:]:
    sock.connect(arg)

while True:
    message = sock.recv()
    print message


"""
Boadbalancer

python pushpull_puller.py tcp://localhost:8080 tcp://localhost:8081
python pushpull_puller.py tcp://localhost:8080 tcp://localhost:8081
"""
