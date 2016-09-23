"""
Radial Bias Function (RBF), or gaussian kernel function.

f(x, y, r) = exp(- r * ||x - y||_2^2)
"""

import numpy as np
import chainer.functions as F
from chainer import cuda, Function, gradient_check, report, training, utils, Variable

batch = 32
dim = 100
x_data = np.random.rand(batch, dim).astype(np.float32)
y_data = np.random.rand(batch, dim).astype(np.float32)
r_init_data = np.random.rand(batch, dim).astype(np.float32)

x = Variable(x_data)
y = Variable(y_data)
r = Variable(r_init_data)

# Forward
print("# Forward")
z = F.exp(- F.sum(r * (x - y) ** 2, axis=1))
print(z.data.shape)

# Backward
print("# Backward")
z.grad = np.random.rand(batch).astype(np.float32)
print("r.grad")
print(r.grad)
z.backward()

# Show
print("r.grad")
print(r.grad)



