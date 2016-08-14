import __future__

import logging
from multiprocessing import Process
from multiprocessing import Queue
import os
from random import random
import random
import scipy.io
import signal
from threading import Thread

import numpy as np
from preprocessor import PreProcessor


class M():
    TERMINATE = "terminate"
    STOP = "stop"
    READ = "read"

class DataReader(object):
    """Data Reader reads batch-data
    """
    
    TIME_WAIT = 60 * 10
    N_READ_WORKERS = 3
    N_PREFETCHES = N_READ_WORKERS
    N_SHUFFLE = 3
    
    def __init__(self, de_context, preprocessor=PreProcessor()):
        """
        Parameters
        ----------
        de_context: DeContext
        _preprocessor: PreProcessor
        
        Attributes
        ----------
        _train_data_location: str
            training data location including schema
        _valid_data_location: str
            training data location including schema
        _preprocessor: PreProcessor
        _dp_queue: Queue
            data pointer queue
        _data_queue: Queue
            data queue, which holds mini-batch data
        _m_queue: Queue
            message queue
        _n_epoch: int
        _batch_size: int
            batch size
        read_workers: List of Process
            read_workers read data from _dp_queue
        N_READ_WORKERS: int
        N_SHUFFLE: int
        """
        
        # Attributes
        self._train_data_location = de_context.data_config.train_data_location
        self._valid_data_location = de_context.data_config.valid_data_location
        
        self._preprocessor = preprocessor
        self._n_epoch = de_context.global_config.n_epoch
        self._batch_size = de_context.global_config.batch_size
        
        self._dp_queue = Queue(0)
        self._data_queue = Queue(0)
        self._m_queue = Queue()
        self.read_workers = []
        
        # Put data pointers
        self._put_all_data_pointers()
        
        # Create and run workers
        self._create_and_run_read_workers()
        
        # Set signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    
    def _create_and_run_read_workers(self):
        for i in range(self.N_READ_WORKERS):
            read_worker = ReadWorker(self._train_data_location,
                                     self._batch_size,
                                     self._dp_queue, 
                                     self._data_queue,
                                     self._m_queue) 

            read_worker.start()
            self.read_workers.append(read_worker)

            # Send read message
            self._m_queue.put(M.READ)
        
    def _put_all_data_pointers(self):
        fnames = os.listdir(self._train_data_location)
        logging.warn("Put all data pointers {}..{}".format(fnames[0:5], fnames[-5:]))
        for _ in range(self.N_SHUFFLE):
            random.shuffle(fnames)
            for fname in fnames:
                self._dp_queue.put(fname)
        
    @property
    def n_iter(self):
        fnames = os.listdir(self._train_data_location)
        return self._n_epoch * len(fnames) 
        
    def read(self, ):
        """Read mini-batch data from _train_data_location and pre process these data
        
        """
        data_ = self._data_queue.get(self.TIME_WAIT)
        data = self._preprocessor.process(data_)
        
        # Send read message
        self._m_queue.put(M.READ)
        
        return data
    
    def terminate_workers(self):
        for w in self.read_workers: w.terminate()
        
    def stop_workers(self):
        for _ in range(self.N_READ_WORKERS): self._m_queue.put(M.STOP)
    
    def _signal_handler(self, signal, frame):
        self.terminate_workers()
    
class ReadWorker(Process):
    
    def __init__(self, train_data_location, batch_size, 
                 dp_queue, data_queue, m_queue):
        """
        Parameters
        ----------
        
        train_data_location: str
            training data location including schema
        batch_size: int
        dp_queue: Queue
            data pointer queue
        data_queue: Queue
            data pointer queue
        m_queue: Queue
            message queue
            
        Attributes
        ----------
        _train_data_location: str
            training data location including schema
        _dp_queue: Queue
            data pointer queue
        _data_queue: Queue
            data pointer queue
        _m_queue: Queue
            message queue
        _remaining_data: dict
            a mat file format, a key represents feature(s) and label(s)
        """
        
        super(ReadWorker, self).__init__()
        
        self._train_data_location = train_data_location
        self._batch_size = batch_size
        self._dp_queue = dp_queue 
        self._data_queue = data_queue
        self._m_queue = m_queue
        self._remaining_data = {}
        
    def _read_data(self, ):
        """Read data
        
        This method is called in a worker process
        """
        while True:
            # Block
            m = self._m_queue.get()

            if m == M.STOP:
                # Stop
                break 
            
            if m == M.READ:
                # Read data of batch size
                data = self._read_data_of_batch_size() 
            
                # Put
                self._data_queue.put(data)
            
    def _read_data_of_batch_size(self, ):

        while True:
            rdata = self._get_read_put()  # read data
            cdata = self._cumulate_to_remaining_data(rdata)  # cumulated data

            # greater than batch size
            if self._compare_data_size(cdata) == 1:
                cdata, self._remaining_data = self._separate_to_batch_size_and_rest(cdata)         
                return cdata
            
            # lesser than batch size
            if self._compare_data_size(cdata) == -1:
                self._remaining_data = cdata
                continue
            
            # equal to batch size
            if self._compare_data_size(cdata) == 0:
                self._remaining_data = {}
                return cdata
           
    def _compare_data_size(self, cdata):    
        
        for k in cdata:
            cdata_k = cdata[k]
            if type(cdata_k) == np.ndarray:
                if len(cdata_k) > self._batch_size:
                    return 1
                if len(cdata_k) < self._batch_size:
                    return -1
                if len(cdata_k) == self._batch_size:
                    return 0
            

        raise Exception("Data format is something wrong, NO np.ndarray found.")
     
    def _separate_to_batch_size_and_rest(self, cdata):
        data_of_batch_size = {}
        data_of_rest = {}
        
        for k in cdata:
            cdata_k = cdata[k]
            if type(cdata_k) != np.ndarray:
                continue
            
            data_of_batch_size[k] = cdata_k[:self._batch_size,]
            data_of_rest[k] = cdata_k[self._batch_size:,]
    
        return data_of_batch_size, data_of_rest
    
    def _cumulate_to_remaining_data(self, rdata):

        # empty case  
        if self._remaining_data == {}:
            rdata_ = {}
            for k in rdata:
                rdata_k = rdata[k]
                if type(rdata_k) != np.array:  
                    rdata_[k] = rdata_k 
            return rdata_

        # cumulate
        cdata = {}
        for k in self._remaining_data:
            remaining_data_k = self._remaining_data[k]
            rdata_k = rdata[k]
            
            if type(remaining_data_k) != np.ndarray:
                continue
            
            cdata_k = np.concatenate((remaining_data_k, rdata_k)) 
            cdata[k] = cdata_k
        return cdata

    def _get_read_put(self):
        """
        Return
        ------
        data: dict
            a mat file
        """
        # Get data pointer 
        data_pointer = self._dp_queue.get()
            
        # Read 
        data_path = os.path.join(self._train_data_location, data_pointer)
        data = scipy.io.loadmat(data_path)
        
        # Put data pointer 
        self._dp_queue.put(data_pointer)
        
        return data
                
    def run(self):
        self._read_data()
        
    

