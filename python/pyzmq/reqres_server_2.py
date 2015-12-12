import sys
import zmq
import time
import cPickle as pkl

MAX_MESSAGES = 101

def main():
    host_port = sys.argv[1]
    filepath = sys.argv[2]
    
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.bind(host_port)

    with open(filepath, "w") as fpout:
        while True:
            # throughput
            st = time.time()
            data = sock.recv_pyobj()
            et = time.time()
            recv_elapsed_time = et - st
            
            # latency
            send_st = data["st"]
            latency = et - send_st

            cnt = data["cnt"]
            fpout.write("{}: recv elapsed time: {} [s]\n".format(cnt, recv_elapsed_time))
            fpout.write("{}: latency: {} [s]\n".format(cnt, latency))

            sock.send("{}: OK form {}".format(cnt, host_port))

            if cnt == (MAX_MESSAGES - 1):
                break

if __name__ == '__main__':
    main()
