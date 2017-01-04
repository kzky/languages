import time
import zlib
import snappy
import marshal
import numpy as np
import scipy.stats
import sys
#import numexpr as ne
import cPickle as pickle
#from itertools import accumulate
#import pyximport; pyximport.install()
#import cumsum

def create_grad(size):
    return np.random.randn(size)

def sparsize(x, p):
    p = p / 2
    q = np.abs(-scipy.stats.norm.ppf(p))
    pos_idx = np.where(x > q)[0]
    neg_idx = np.where(x < -q)[0]
    return np.concatenate((pos_idx, neg_idx))

def serialize(x):
    return marshal.dumps(x)

def compress(x):
    return snappy.compress(x)

def deserialize(x):
    return marshal.loads(x)

def decompress(x):
    return snappy.decompress(x)

def compute_diff_index(x):
    return x[1:] - x[0:-1]

def normal_no_comp(max_index=int(2 * 1e6), p=0.07):
    print("### Normal w/o Compression ###")
    
    # Settings
    size = int(max_index)
    x = create_grad(size)
    elapsed_times = []

    # Sparsize
    st = time.time()
    x = sparsize(x, p)
    et = time.time() - st
    elapsed_times.append(et)
    print("Sparsize:{}[s]".format(et))

    # Serialize
    st = time.time()
    x_ser = serialize(x.tolist())
    et = time.time() - st
    elapsed_times.append(et)
    print("Serialize:{}[s],{}[len],{}[B]".format(et, len(x_ser), sys.getsizeof(x_ser)))

    print("Total(Sparse+Ser):{}[s]".format(np.sum(elapsed_times)))

    elapsed_times = []
    # Deserialize
    st = time.time()
    xdeser = deserialize(x_ser)
    et = time.time() - st
    elapsed_times.append(et)
    print("Deserialize:{}[s]".format(et))
    
    print("Total(Decomp+Deser):{}[s]".format(np.sum(elapsed_times)))

def normal(max_index=int(2 * 1e6), p=0.07):
    print("### Normal ###")
    
    # Settings
    size = int(max_index)
    x = create_grad(size)
    elapsed_times = []

    # Sparsize
    st = time.time()
    x = sparsize(x, p)
    et = time.time() - st
    elapsed_times.append(et)
    print("Sparsize:{}[s]".format(et))

    # Serialize
    st = time.time()
    x_ser = serialize(x.tolist())
    et = time.time() - st
    elapsed_times.append(et)
    print("Serialize:{}[s],{}[len],{}[B]".format(et, len(x_ser), sys.getsizeof(x_ser)))

    # Compress
    st = time.time()
    x_ser_comp = compress(x_ser)
    et = time.time() - st
    elapsed_times.append(et)
    print("Compress:{}[s],{}[len],{}[B]".format(et, len(x_ser_comp), sys.getsizeof(x_ser_comp)))
    print("Total(Sparse+Ser+Comp):{}[s]".format(np.sum(elapsed_times)))
    elapsed_times = []

    # Decompress
    st = time.time()
    xdecomp = decompress(x_ser_comp)
    et = time.time() - st
    elapsed_times.append(et)
    print("Decompress:{}[s]".format(et))

    # Deserialize
    st = time.time()
    xdeser = deserialize(xdecomp)
    et = time.time() - st
    elapsed_times.append(et)
    print("Deserialize:{}[s]".format(et))
    
    print("Total(Decomp+Deser):{}[s]".format(np.sum(elapsed_times)))

def diff_index(max_index=int(2 * 1e6), p=0.07):
    print("### Diff Index ###")
    
    # Settings
    size = int(max_index)
    x = create_grad(size)
    elapsed_times = []

    # Sparsize
    st = time.time()
    x = sparsize(x, p)
    et = time.time() - st
    elapsed_times.append(et)
    print("Sparsize:{}[s]".format(et))

    # Diff Index
    st = time.time()
    x_diff = compute_diff_index(x)
    et = time.time() - st
    elapsed_times.append(et)
    print("DiffIndex:{}[s]".format(et))

    # Serialize
    st = time.time()
    x_diff_ser = serialize([int(x[0])] + x_diff.tolist())  # need copy [int(x[0])]
    et = time.time() - st
    elapsed_times.append(et)
    print("Serialize:{}[s],{}[len],{}[B]".format(et, len(x_diff_ser), sys.getsizeof(x_diff_ser)))

    # Compress
    st = time.time()
    x_diff_ser_comp = compress(x_diff_ser)
    et = time.time() - st
    elapsed_times.append(et)
    print("Compress:{}[s],{}[len],{}[B]".format(et, len(x_diff_ser_comp), sys.getsizeof(x_diff_ser_comp)))
    print("Total(Sparse+Diff+Ser+Comp):{}[s]".format(np.sum(elapsed_times)))
    elapsed_times = []

    # Decompress
    st = time.time()
    x_diff_decomp = decompress(x_diff_ser_comp)
    et = time.time() - st
    elapsed_times.append(et)
    print("Decompress:{}[s]".format(et))

    # Deserialize
    st = time.time()
    x_diff_deser = deserialize(x_diff_decomp)
    et = time.time() - st
    elapsed_times.append(et)
    print("Deserialize:{}[s]".format(et))
    
    # Cumsum
    st = time.time()
    #x = reduce(lambda x, y: x+y, x_diff_deser)
    x = np.cumsum(x_diff_deser)
    #x = np.add.accumulate(x_diff_deser)
    #x = np.array(x_diff_deser)
    #for i in range(len(x) -1):
    #    x[i+1] += x[i]
    #x = np.array(x_diff_deser)
    #cumsum.cumsum(x, len(x))
    #x = np.array(x_diff_deser)
        
    et = time.time() - st
    elapsed_times.append(et)
    print("Cumsum:{}[s]".format(et))
    print("Total(Decomp+Deser+Cumsum):{}[s]".format(np.sum(elapsed_times)))


if __name__ == '__main__':
    max_index = int(0.1 * 1e6)
    p = 0.15
    normal_no_comp(max_index=max_index, p=p)
    normal(max_index=max_index, p=p)
    diff_index(max_index=max_index, p=p)

