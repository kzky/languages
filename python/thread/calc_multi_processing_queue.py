from multiprocessing import Process, Queue
from multiprocessing.queues import SimpleQueue
import numpy as np
import time

def f(q_inp, q_out):
    data = q_inp.get()
    st = data["st"]
    et = time.time()
    print("Chilld:{}[s]".format(et - st))
    st = time.time()
    data["st"] = st
    q_out.put(data)

if __name__ == '__main__':
    data = {
        "x": np.random.rand(64, 3, 224, 224),
        #"y": np.random.rand(128, 1000),
        #"y": np.random.rand(64,),
    }
    q_inp = Queue()
    q_out = Queue()

    st = time.time()
    data["st"] = st
    q_inp.put(data)
    p = Process(target=f, args=(q_inp, q_out))
    p.start()
    data = q_out.get()
    p.join()
    st = data["st"]
    et = time.time()
    print("Parent:{}[s]".format(et - st))
    
    p.terminate()
