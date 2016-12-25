import numpy as np
import chainer
import chainer.variable as variable
from chainer.functions.activation import lstm
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer.cuda import cupy as cp
from chainer import datasets, iterators, optimizers, serializers, function_hooks
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from collections import OrderedDict
import logging
import time
from vaegan import mlp_models
import chainer, chainer.functions as F, chainer.links as L
    
class MLP(Chain):
    def __init__(self, ):
        super(MLP, self).__init__(
            l0=L.Linear(100, 50),
            l1=L.Linear(50, 10),
        )
        
    def __call__(self, x):
        h = F.relu(self.l0(x))
        h = self.l1(h)
        return h

def main():
    model = MLP()
    optimizer = optimizers.Adam(0.01)
    optimizer.setup(model)
    optimizer.use_cleargrads()

    bs = 8
    x_data = np.random.randn(bs, 100).astype(np.float32)
    x = Variable(x_data)
    y = model(x)
    z = F.sum(y)

    with function_hooks.PrintHook():
        z.backward()
        

if __name__ == '__main__':
    main()




    
