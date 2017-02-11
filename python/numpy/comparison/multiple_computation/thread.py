from threading import Thread
import numpy as np
import time

class Worker(Thread):
    def __init__(self, ):
        super(Worker, self).__init__()

        shape = (1000, 1000)
        self.data = np.random.rand(*shape)
        self.threshold = 1
        
    def run(self, ):
        data = self.data
        for i in range(self.threshold):
            c = np.dot(data, data)

def main():
    # Initialize
    n_workers = 4
    workers = []
    for i in range(n_workers):
        worker = Worker()
        workers.append(worker)
        
    # Start
    st = time.time()
    for i in range(n_workers):
        workers[i].start()

    # Joins
    for i in range(n_workers):
        workers[i].join()

    et = time.time() - st
    print("ElapsedTime:{}[s]".format(et))

if __name__ == '__main__':
    main()
    
