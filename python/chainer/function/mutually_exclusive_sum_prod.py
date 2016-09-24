"""
Mutually exclusive Sum-Prod with given axis

\sum_{k=1}^{K} o_k \prod_{k=1, k \neq j}^{K} (1 - o_k)
"""

import numpy as np
import chainer.functions as F
from chainer import cuda, Function, gradient_check, report, training, utils, Variable

batch = 32
dim = 10

o_data = np.random.rand(batch, dim).astype(np.float32)
o = Variable(o_data)

#TODO: find more computationally efficient way.
# Forward
print("# Forward")
sum_prod = 0
for i in range(dim):
    indices = np.asarray([j for j in range(dim) if j != i ], dtype=np.int32)
    prod = 1
    for j in indices:
        prod *= (1 - o[:, j])
    sum_prod += o[:, i] * prod

print(sum_prod.data.shape)

# Backward
print("# Backward")
sum_prod.grad = np.random.rand(*sum_prod.data.shape).astype(np.float32)
sum_prod.backward()
print(o.grad)
print(o.grad.shape)

    
