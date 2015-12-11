import sys
import zmq

context = zmq.Context()
sock = context.socket(zmq.SUB)
sock.setsockopt(zmq.SUBSCRIBE, '')

for arg in sys.argv[1:]:
    sock.connect(arg)

while True:
    message = sock.recv()
    print message

"""
Broadcast

python pubsub_subscriber_3.py tcp://localhost:8080
python pubsub_subscriber_3.py tcp://localhost:8080
"""
