from multiprocessing import Process, Queue
from multiprocessing import sharedctypes
import numpy as np
import time
from numpy import ctypeslib

def f(q_in, q_out):
    x = np.random.rand(128* 3 * 224 * 224)
    
    S = q_in.get()
    x_ = ctypeslib.as_array(S.get_obj())
    
    st = time.time()
    x_[:] = x
    x_[0] = 1000
    data = {"x": x_, "st": st}
    q_out.put(data)

if __name__ == '__main__':
    buff = np.random.rand(128* 3* 224* 224)
    S = sharedctypes.Array("f", buff)
    q_in = Queue()
    q_out = Queue()

    #RuntimeError: SynchronizedArray objects should only be shared between processes through inheritance

    p = Process(target=f, args=(q_in, q_out))
    p.start()
    q_in.put(S)
    data = q_out.get()

    print("Parent:{}[s]".format(time.time() - data["st"]))
    x_ = ctypeslib.as_array(data["x"].get_obj())
    print("init-val:{}, changed-val:{}".format(buff[0], x_[0]))
    p.terminate()
