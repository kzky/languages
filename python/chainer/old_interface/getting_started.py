#!/usr/bin/env python

import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F

# basic
print "# basic"
x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
print x
print x.data

y = x**2 - 2 * x + 1
print y
print y.data

y.backward()
print x.grad

# intermediate grad
print "# intermediate grad"
z = 2 * x
y = z**2 - 2 * z + 1
y.backward(retain_grad=False)
print z.grad

z = 2 * x
y = z**2 - 2 * z + 1
y.backward(retain_grad=True)
print z.grad

# multi element array (ndarray where n >=2)
print "# multi element array (ndarray where n >=2)"
x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = x**2 - 2*x + 1
y.backward()
print x.grad

x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = x**2 - 2*x + 1
y.grad = np.ones((2, 3), dtype=np.float32) # have to set init grad values
y.backward()
print x.grad

# parameterized function
print "# parameterized function"
f = F.Linear(3, 2)
print f.W
print f.b

x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = f(x)
print y.data

y.grad = np.ones((2, 2), dtype=np.float32)  # have to set init grad values

y.backward()
print f.gW, f.gb
y.backward()
print f.gW, f.gb

f.gW.fill(0), f.gb.fill(0)  # have to fill
y.backward()
print f.gW, f.gb

# function set
print "# function set"
model = FunctionSet(
    l1=F.Linear(4, 3),
    l2=F.Linear(3, 2),
)
print model
model.l3 = F.Linear(2, 2)

x = Variable(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32))
h1 = model.l1(x)
h2 = model.l2(h1)
h3 = model.l3(h2)

# have to set init grad values and fill grads, which will be done using optimizer
print model.parameters
print model.gradients
   
   
