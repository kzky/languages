from multiprocessing import Process, Queue
from multiprocessing import sharedctypes
import numpy as np
import time
from numpy import ctypeslib

n_params = 100 * 1000

def f(q, S):
    x = np.random.rand(n_params)
    x_ = ctypeslib.as_array(S.get_obj())

    st = time.time()
    #x_ = x  # This does not overwrite the shared memory.
    st0 = time.time()
    x_[:] = x
    print("SetTime:{}".format(time.time() - st0))
    x_[0] = 1000
    data = {"x": None, "st": st}
    q.put(data)

if __name__ == '__main__':
    buff = np.random.rand(n_params)
    S = sharedctypes.Array("f", buff)
    q = Queue()
    p = Process(target=f, args=(q, S, ))
    p.start()
    data = q.get()

    print("Parent:{}[s]".format(time.time() - data["st"]))
    x_ = ctypeslib.as_array(S.get_obj())
    print("Shape:{}".format(x_.shape))
    print("init-val:{}, changed-val:{}".format(buff[0], x_[0]))
    p.terminate()
