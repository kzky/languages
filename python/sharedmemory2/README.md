# Shared Memory for Data Storage

When reading data in python with some preprocessing, threading works very bad due to GIL, so Multiprocessing is the natual choice. However, if the data is large enough and we need IPC over Queue for the main process having GPU, this IPC becomes overhead, it takes about 300[ms] based on this prelimilary experiment. To circumvent this, we can use SharedMemory. I did the following expeirments. 

## Experiment Settings
- Use multiprocessing.sharedctypes.Array as SharedMermoy
-- I would like to use it as a data store, Shared Memory should have R/W lock.
- Assume data is already read in the memory by a worker process.
- Shape are known.
- It will be fed into SharedMemroy and the main thread read from it.
- As comparison, use IPC with multiprocess.Queue.


## Results

[IPC](data_with_queue.py)
```
$ python data_with_queue.py   
Parent:0.445959091187[s]
```

[SharedMem](data_with_shmarray.py)
```
$ python data_with_shmarray.py
/usr/local/lib/python2.7/dist-packages/numpy/ctypeslib.py:435: RuntimeWarning: Item size computed from the PEP 3118 buffer format string does not match the actual item size.
  return array(obj, copy=False)
Parent:0.0340781211853[s]
/usr/local/lib/python2.7/dist-packages/numpy/ctypeslib.py:435: RuntimeWarning: Item size computed from the PEP 3118 buffer format string does not match the actual item size.
  return array(obj, copy=False)
init-val:0.102639386732, changed-val:1000.0
```

## Discussion
SahredArray is 10-fold fast! If data shape is known in advance, it should be used; otherwise, secure the large memory in advance, then do some logic to use that memroy region efficiently.






