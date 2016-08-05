# Introduction

This introduction is based on [this page](http://docs.chainer.org/en/stable/tutorial/index.html)

Core concept of Chainer is *Define-by-Run*. When the computational graph created, forward computation also runs at the same time, so that it is called *Define-by-Run*, or referred to as imperative.

# Main Components
### Variable
- Namely, a variable.
- It can call backward function, note gradients are not overwritten, but accumulated.
- For multi-element array, set the initial values for the gradient of the last variable which calls the backward function.
- Gradients are stored in the w.r.t.-variables, e.g., y = 2*x, dy/dx is stored in x.grad.

### Link
- Parameterized function, where the parameters are optimized.
- Parameters of a function is hidden in Link, so this is a high-level api.
- The low-level corresponding api, for e.g., Linaer, is linear, which tasks a, W, b.

### Chain
- To make a model (DNN) as a class, inherite chain. it support parameter management, CPU/GPU migration support, robust and flexible save/load features, etc.
- Usually, forward computation is coded in the *__call__* method.
- Chain is the child class of Link.
- ChainList can hold the arbitrary number of Link, if the number of links is fixed, simply use Chain, and it namely seems like a list of links in order to add a link at the end of ChainList.
- A typical usage of Chain is the following, pass paramerized functions to *super.__init__* and implement forward pass in *__call__*.

```python
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(4, 3),
            l2=L.Linear(3, 2)
        )

    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)
```

### Optimizer
- Optimizer takes model (chain) and compute gradients.
- We can add hook function after computatin gradient and before updating parameters.
- Low-level api for optimizing.
- A typical usage is the following (as far as I remember based on the previsou docs)

```python
optimizer = optimizers.SGD()
optimizer.setup(obj_func)
while ...:  ## data-loop
    models.zerograds()  # due to gradients are accumulated.
    obj_func.backward()
    optimizer.update()
```

### Trainer
- Class in order to make the training-loop easy to implement.
- It can be used with datasets and iterators moduels and also with Extension class.

### Serializer
- It can save Link, Optimizer, Trainer as npz or hdf5
- Some attributes, e.g., parameters  are serialized automatically, the others are not. For such attributes to be saved, we have to specify by calling Link.add_persistent() method.

### Example: Multi-layer Perceptron on MNIST

This is a hello world example in Deep Learning.

See the code [here]().

# Review
When comparing with previous versions where only low-level APIs are published without high-level APIs, it became confusing to me, especillay the optimization loop, Trainer, Updater, Evaluator, *Report, etc, which is normally out of scope of minimum components of what deep learning library is.

# Reference
- http://docs.chainer.org/en/stable/tutorial/basic.html
- http://docs.chainer.org/en/stable/tutorial/index.html

