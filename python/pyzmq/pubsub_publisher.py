import zmq
import time
from random import randrange

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

while True:
    zipcode = randrange(1, 100000)
    temperature = randrange(-80, 135)
    relhumidity = randrange(10, 60)

    string = "%i %i %i" % (zipcode, temperature, relhumidity)
    print string

    #time.sleep(1)
    socket.send_string(string)
