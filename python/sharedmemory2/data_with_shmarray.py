from multiprocessing import Process, Queue
from multiprocessing import sharedctypes
import numpy as np
import time
from numpy import ctypeslib

def f(q, S):
    x = np.random.rand(128* 3 * 224 * 224)
    x_ = ctypeslib.as_array(S.get_obj())

    st = time.time()
    #x_ = x
    x_[:] = x
    x_[0] = 1000
    data = {"x": None, "st": st}
    q.put(data)

if __name__ == '__main__':
    buff = np.random.rand(128* 3* 224* 224)
    S = sharedctypes.Array("f", buff)
    q = Queue()
    p = Process(target=f, args=(q, S, ))
    p.start()
    data = q.get()

    print("Parent:{}[s]".format(time.time() - data["st"]))
    x_ = ctypeslib.as_array(S.get_obj())
    print("init-val:{}, changed-val:{}".format(buff[0], x_[0]))
    p.terminate()
