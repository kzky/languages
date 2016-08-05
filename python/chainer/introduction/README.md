# Introduction

This introduction is based on [this page](http://docs.chainer.org/en/stable/tutorial/index.html)

Core concept of Chainer is *Define-by-Run*. When the computational graph created, forward computation also runs at the same time, so that it is called *Define-by-Run*, or referred to as imperative.

# Installation

See [this](https://github.com/pfnet/chainer) for enabling cudnn. For ubuntu16.04, take care to use gcc-4.9 and g++-4.0, also the environment variables.

# Main Components
### Variable
- Namely, a variable.
- It can call backward function, note gradients are not overwritten, but accumulated.
- For multi-element array, set the initial values for the gradient of the last variable which calls the backward function.
- Gradients are stored in the w.r.t.-variables, e.g., y = 2*x, dy/dx is stored in x.grad.
- Usually, Chainer releases intermediate gradients. To hold, pass retain_grad=True to backward.

### Link
- Parameterized function, where the parameters are optimized.
- Parameters of a function is hidden in Link, so this is a high-level api.
- The low-level corresponding api, for e.g., Linaer, is linear, which tasks a, W, b.

### Chain
- To make a model (DNN) as a class, inherite chain. it support parameter management, CPU/GPU migration support, robust and flexible save/load features, etc.
- Usually, forward computation is coded in the *\_\_call\_\_* method.
- Chain is the child class of Link.
- ChainList can hold the arbitrary number of Link, if the number of links is fixed, simply use Chain, and it namely seems like a list of links in order to add a link at the end of ChainList.
- A typical usage of Chain is the following, pass paramerized functions to *super.__init__* and implement forward pass in *\_\_call\_\_*.

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
When comparing with previous versions where only low-level APIs are published without high-level APIs, it became confusing to me, especillay the optimization loop, Trainer, Updater, Evaluator, *Report, etc.

# Note

## gcc, g++, cpp

Use 4.9 with CUDA 7.5 before installing chainer.

## Device ids are reverted or randomed

The following is the result of *nvidia-smi* on my ubuntu16.04.

```sh
Fri Aug  5 21:13:15 2016
+------------------------------------------------------+
| NVIDIA-SMI 352.79     Driver Version: 352.79         |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GT 610      Off  | 0000:01:00.0     N/A |                  N/A |
| 40%   36C    P8    N/A /  N/A |    196MiB /  1023MiB |     N/A      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 780     Off  | 0000:02:00.0     N/A |                  N/A |
| 30%   39C    P8    N/A /  N/A |     11MiB /  3071MiB |     N/A      Default |
+-------------------------------+----------------------+----------------------+
```

When I set device\_id=1, I got the error "CUDNN\_STATUS\_ARCH\_MISMATCH", which indicates the compute capability is mismatching so that a user can not use the CUDNN in my case; however, I set devide\_id=0, my code can run.

Ispecting more with "chainer.cuda.Device(device\_id)" shows.

```python
In [1]: import chainer.cuda

In [2]: d0 = chainer.cuda.Device(0)                                                                                                           

In [3]: d0.compute_capability
Out[3]: '35'

In [4]: d1 = chainer.cuda.Device(1)                                                                                                           

In [5]: d1.compute_capability
Out[5]: '21'
```

Comparing the "nvidia-smi" results with this result indicates the difference between device ids of nvidia-smi and ones indexed by Chainer because [Wikipedia](https://en.wikipedia.org/wiki/CUDA) says that GTX 750 has cc=35 and GeForce GT 610 has cc=21.

# Reference
- http://docs.chainer.org/en/stable/tutorial/basic.html
- http://docs.chainer.org/en/stable/tutorial/gpu.html#run-neural-networks-on-a-single-gpu
