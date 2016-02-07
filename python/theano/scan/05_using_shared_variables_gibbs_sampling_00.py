import theano
import numpy as np
from theano import tensor as T

# Paramter initial values
W_values = np.random.rand(10, 5).astype(np.float32)
bvis_values = np.random.rand(10).astype(np.float32)
bhid_values = np.random.rand(10).astype(np.float32)

# Paramter symbols
W = theano.shared(W_values)
bvis = theano.shared(bvis_values)
bhid = theano.shared(bhid_values)

trng = T.shared_randomstreams.RandomStreams(1234)

def OneStep(vsample):
    hmean = T.nnet.sigmoid(theano.dot(vsample, W) + bhid)
    hsample = trng.binomial(size=hmean.shape, n=1, p=hmean)
    vmean = T.nnet.sigmoid(theano.dot(hsample, W.T) + bvis)
    return trng.binomial(size=vsample.shape, n=1, p=vmean,
                         dtype=theano.config.floatX)

sample = theano.tensor.vector()

values, updates = theano.scan(OneStep, outputs_info=sample, n_steps=10)

gibbs10 = theano.function([sample], values[-1], updates=updates)

print gibbs10([np.random.rand(5)])
