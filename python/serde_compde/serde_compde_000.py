import time
import zlib
import snappy
import marshal
import numpy as np


def normal():
    print("### Normal ###")
    
    # Settings
    p = 0.15  # density
    max_index = int(10 * 1e6)
    size = int(max_index * p)
    x = np.sort(np.random.choice(max_index, size, replace=False))
    elapsed_times = []

    # Serialize
    st = time.time()
    x_ser = marshal.dumps(x)
    et = time.time() - st
    elapsed_times.append(et)
    print("Serialize:{}[s],{}[len]".format(et, len(x_ser)))

    # Compress
    st = time.time()
    x_ser_comp = snappy.compress(x_ser)
    et = time.time() - st
    elapsed_times.append(et)
    print("Compress:{}[s],{}[len]".format(et, len(x_ser_comp)))

    print("Total(Ser+Comp):{}[s]".format(np.sum(elapsed_times)))
    elapsed_times = []

    # Decompress
    st = time.time()
    xdecomp = snappy.decompress(x_ser_comp)
    et = time.time() - st
    elapsed_times.append(et)
    print("Decompress:{}[s]".format(et))

    # Deserialize
    st = time.time()
    xdeser = marshal.loads(xdecomp)
    et = time.time() - st
    elapsed_times.append(et)
    print("Deserialize:{}[s]".format(et))
    
    print("Total(Decomp+Deser):{}[s]".format(np.sum(elapsed_times)))

def diff_index():
    print("### Diff Index ###")
    
    # Settings
    p = 0.15  # density
    max_index = int(10 * 1e6)
    size = int(max_index * p)
    x = np.sort(np.random.choice(max_index, size, replace=False))
    elapsed_times = []

    # Diff Index
    st = time.time()
    x_diff = x[1:] - x[0:-1]
    et = time.time() - st
    elapsed_times.append(et)
    print("DiffIndex:{}[s]".format(et))
    
    # Serialize
    st = time.time()
    x_diff_ = [int(x[0])] + x_diff.tolist()
    x_diff_ser = marshal.dumps(x_diff_)
    et = time.time() - st
    elapsed_times.append(et)
    print("Serialize:{}[s],{}[s]".format(et, len(x_diff_ser)))

    # Compress
    st = time.time()
    x_diff_ser_comp = snappy.compress(x_diff_ser)
    et = time.time() - st
    elapsed_times.append(et)
    print("Compress:{}[s],{}[len]".format(et, len(x_diff_ser_comp)))

    print("Total(Diffindex+Ser+Comp):{}[s]".format(np.sum(elapsed_times)))
    elapsed_times = []

    # Decompress
    st = time.time()
    x_diff_decomp = snappy.decompress(x_diff_ser_comp)
    et = time.time() - st
    elapsed_times.append(et)
    print("Decompress:{}[s]".format(et))

    # Deserialize
    st = time.time()
    x_diff_deser = marshal.loads(x_diff_decomp)
    et = time.time() - st
    elapsed_times.append(et)
    print("Deserialize:{}[s]".format(et))
    
    # Cumsum
    st = time.time()
    x = reduce(lambda x, y: x+y, x_diff_deser)
    et = time.time() - st
    elapsed_times.append(et)
    print("Cumsum:{}[s]".format(et))
    print("Total(Decomp+Deser+Cumsum):{}[s]".format(np.sum(elapsed_times)))


if __name__ == '__main__':
    normal()
    diff_index()


