import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)

# Forward
print("# Forward")
y = x**2 - 2 * x + 1
print(y.data)

# Backward
print("# Backward")
y.backward()
print(y.grad)
print(x.grad)
y.backward()
print(x.grad)  ## gradient is accumulated

# Forward/Backward with retain_grad
print("Forward/Backward with retain_grad")
z = 2 * x
y = x**2 - z + 1
y.backward(retain_grad=True)
print(z.grad)

# Forward/Backward with multi-element array
print("# Forward/Backward with multi-element array, we must set init values for grad of the last variable")
x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = x**2 - 2 * x + 1
y.grad = np.ones((2, 3), dtype=np.float32)
y.backward()
print(x.grad)

# Link (e.g., linear)
print("# Link (e.g., linear)")
x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
print("shape of x is {}".format(x.data.shape))
f = L.Linear(3, 2)
y = f(x)
f.zerograds()
y.grad = np.ones((2, 2), dtype=np.float32)
y.backward()
print(f.W.grad)
print(f.b.grad)

# Write a model as a chain
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(4, 3),
            l2=L.Linear(3, 2)
        )

    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)

# Optimizer
model = MyChain()
optimizer = optimizers.SGD()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

## One way to use in the training loop
###model.zerograds()
#### call backward for objective function.
###optimizer.update()

## The other way to use
def lossfunc():
    loss = None
    # ...
    return loss

###optimizer.update(lossfunc, args)

# Trainer
## ...

