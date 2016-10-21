from multiprocessing import Process, Queue
from multiprocessing.queues import SimpleQueue

import numpy as np
import time

def f(q):
    x = np.random.rand(128, 3, 224, 224)
    st = time.time()
    data = {"x": x, "st": st}
    q.put(data)

if __name__ == '__main__':

    q = Queue()
    p = Process(target=f, args=(q, ))
    p.start()
    data = q.get()
    print("Parent:{}[s]".format(time.time() - data["st"]))
    p.terminate()
