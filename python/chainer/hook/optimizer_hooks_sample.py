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

def sample_hook(optimizer):
    for p in optimizer.target.params():
        print(type(p))
        print(p.data)
        print(p.grad)

def grad_norm_hook(optimizer):
    for p in optimizer.target.params():
        grad_data = p.grad
        shape = grad_data.shape
        reshape = (1, np.prod(shape), )

        grad = Variable(grad_data)
        grad_reshape = F.reshape(grad, reshape)
        grad_norm = F.normalize(grad_reshape)
        grad_norm_reshape = F.reshape(grad_norm, shape)

        p.grad = grad_norm_reshape.data
        print(np.linalg.norm(grad_norm.data))
        
def hello():
    model = MLP()
    optimizer = optimizers.SGD(1)
    optimizer.setup(model)
    optimizer.use_cleargrads()
    optimizer.add_hook(sample_hook, name="sample_hook")

    bs = 8
    x_data = np.random.randn(bs, 100).astype(np.float32)
    x = Variable(x_data)
    y_data = np.random.randn(bs).astype(np.int32)
    y_t = Variable(y_data)
    y = model(x)
    l = F.softmax_cross_entropy(y, y_t)
    model.cleargrads()
    l.backward()
    optimizer.call_hooks()    
    optimizer.update()    
    optimizer.call_hooks()  # sanity check for data and grad are the same
    
def grad_norm():
    model = MLP()
    optimizer = optimizers.SGD(0.01)
    optimizer.setup(model)
    optimizer.use_cleargrads()
    optimizer.add_hook(grad_norm_hook, name="grad_norm_hook")

    bs = 8
    x_data = np.random.randn(bs, 100).astype(np.float32)
    x = Variable(x_data)
    y_data = np.random.randn(bs).astype(np.int32)
    y_t = Variable(y_data)
    y = model(x)
    l = F.softmax_cross_entropy(y, y_t)
    model.cleargrads()
    l.backward()
    optimizer.call_hooks()    
    optimizer.update()    

            
if __name__ == '__main__':
    #hello()
    grad_norm()

