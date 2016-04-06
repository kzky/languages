#!/usr/bin/python
# -*- coding: utf-8 -*-

# url
## http://ja.pymotw.com/2/threading/
## http://docs.python.jp/2.7/library/threading.html

import threading
import glob
import time
import Queue
import cv2
import pickle

# http://docs.python.jp/2/library/queue.html#module-Queue

# read pickle to ndarray
def read_pkl_2_ndarray(filepath):
    
    return pickle.load(open(filepath, "r"))

# queue
## targetにinstanceを渡せない
## 渡したいなら，Threadクラスを継承したクラスを作る
queue = Queue.Queue()

# wokrer
def worker():
    """
    """
    while True:
        path = queue.get()
        read_pkl_2_ndarray(path)
        queue.task_done()
        pass
    pass

# put task
base_dirpath = "/home/kzk/datasets/cifar10/train_pkl"
filepaths = glob.glob("{}/*".format(base_dirpath))
for filepath in filepaths:
    queue.put(filepath)
    pass

# create/start  thread
st = time.time()
num_worker_threads = 3
for i in range(num_worker_threads):
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    pass

queue.join()  # wait
et = time.time()
print "total execution time with threading: ", (et - st), "[s]"

# total execution time with threading:  358.308109045 [s] (when in the first time)
# total execution time with threading:  21.4498119354 [s] (when disk cached)


