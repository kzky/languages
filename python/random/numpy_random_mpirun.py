import numpy as np
import multiprocessing as mp
import os

def generate_random_number():
    np.random.seed(10)
    pid = os.getpid()
    y = []
    for i in range(4):
        x = np.random.rand(2)
        y.append(x)
    y = np.asarray(y)
    print(y)
    np.savetxt("generated_data_{}.txt".format(pid), y, delimiter=',')

def main():
    generate_random_number()
        
if __name__ == '__main__':
    main()
    
    
