import sys
import zmq

port = "5560"

# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

print "Collecting updates from server..."

# connect to a server
socket.connect("tcp://localhost:%s" % port)

# filter
topicfilter = "9"
socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

while True:
    string = socket.recv()
    topic, messagedata = string.split()
    print topic, messagedata

    
