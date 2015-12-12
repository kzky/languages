import sys
import time
import zmq
import cPickle as pkl

MAX_MESSAGES = 101

def main():
    host_port = sys.argv[1]
    n = int(sys.argv[2])
    filepath = sys.argv[3]
    
    context = zmq.Context()
    sock = context.socket(zmq.PUB)
    sock.bind(host_port)

    cnt = 0
    with open(filepath, "w") as fpout:
        while (cnt < MAX_MESSAGES):
            # send data
            n_bytes = n * bytearray(1024)
            st = time.time()
            data = {
                "cnt": cnt,
                "st": st,
                "payload": n_bytes
            }
     
            # throughput
            st = time.time()
            sock.send_pyobj(data)
            et = time.time()
            fpout.write("send elapsed time: {} [s]\n".format(et - st))
            cnt += 1

            # prcess something
            time.sleep(0.1)

if __name__ == '__main__':
    main()
