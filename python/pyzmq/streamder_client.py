import time
import zmq

def producer():
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.connect("tcp://127.0.0.1:5559")
    t = time.ctime()
    # Start your result manager and workers before you start your producers
    for num in xrange(200):
        work_message = {'num': "{} at {}".format(num, t)}
        zmq_socket.send_json(work_message)
        #time.sleep(1)

producer()
