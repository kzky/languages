from multiprocessing import Process
import numpy as np
import time

class Worker(Process):
    def __init__(self, iteration):
        super(Worker, self).__init__()

        shape = (1000, 1000)
        self.data = np.random.rand(*shape)
        self.iteration = iteration
        
    def run(self, ):
        data = self.data
        for i in range(self.iteration):
            c = np.dot(data, data)

def compute(n_workers, iteration):
    # Initialize
    n_workers = n_workers
    workers = []
    for i in range(n_workers):
        worker = Worker(iteration)
        workers.append(worker)
        
    # Start
    st = time.time()
    for i in range(n_workers):
        workers[i].start()

    # Joins
    for i in range(n_workers):
        workers[i].join()

    et = time.time() - st
    print("ElapsedTime:{}[s],Nworkers:{},Iter:{}".format(et, n_workers, iteration))

def main():
    n_workers_list = [4, 8, 16, 32]
    iteration_list = [1, 10, 100]

    for n_workers in n_workers_list:
        for iteration in iteration_list:
            compute(n_workers, iteration)        
    
if __name__ == '__main__':
    main()
    

    
    
               
