import numpy as np
import time
import marshal

def main():

    num_layers = 10
    num_param = int(1* 1e6)
    params = []

    for i in range(num_layers):
        params.append((np.random.rand(num_param) * 10).astype(np.int32))

    # tobytes and join
    st = time.time()
    data = []
    for param in params:
        data.append(param.tobytes())
    "".join(data)
    et = time.time() - st
    print("ElapsedTime(tobytes+join):{}[s]".format(et))    
    
    # tolist and marshal
    st = time.time()
    data = []
    for param in params:
        data.append(param.tolist())
    marshal.dumsp(data)
    et = time.time() - st
    print("ElapsedTime(tolist+marshal):{}[s]".format(et))

    
if __name__ == '__main__':
    main()
