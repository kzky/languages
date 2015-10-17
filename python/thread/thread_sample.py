#!/usr/bin/python
# -*- coding: utf-8 -*-

# url
## http://ja.pymotw.com/2/threading/
## http://docs.python.jp/2.7/library/threading.html

import threading
import glob
import time
import zipfile
import Queue

# http://docs.python.jp/2/library/queue.html#module-Queue

# read onley unzip
def unzip(zfin):
    zf = zipfile.ZipFile(zfin, "r")
    for f in zf.namelist():  # zipは複数ファイルが１つにまとめられている前提のため
        #print "unzip", f
        zf.read(f)  # reading only
    zf.close
    pass

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
        unzip(path)
        queue.task_done()
        pass

    pass

# put task
path = "/home/kzk/downloads/*.zip"
for path in glob.glob(path):
    queue.put(path)
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

# total execution time with threading:  60.7147810459 [s]

