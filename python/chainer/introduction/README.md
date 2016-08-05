# Introduction

This introduction is based on [this page](http://docs.chainer.org/en/stable/tutorial/index.html)

Core concept of Chainer is *Define-by-Run*. When the computational graph created, forward computation also runs at the same time, so that it is called *Define-by-Run*, or referred to as imperative.

# Main Components
### Variable

- Namely, a variable.

### Link
- Parameterized function, where the parameters are optimized.
- Parameters of a function is hidden in Link, so this is a high-level api.
- The low-level corresponding api, for e.g., Linaer, is linear, which tasks a, W, b.

### Chain/ChainList
- To make a model (DNN) as a class, inherite chain. it support parameter management, CPU/GPU migration support, robust and flexible save/load features, etc.
- Usually, forward computation is coded in the *__call__* method.
- Chain is the child class of Link
- ChainList can hold the arbitrary number of Link, if the number of links is fixed, simply use Chain.

### Optimizer
- Optimizer takes model (chain) and compute gradients.
- We can add hook function after computatin gradient and before updating parameters.
- Low-level api for optimizing.

### Trainer
- Class in order to make the training-loop easy to implement.
- It can be used with datasets and iterators moduels and also with Extension class.

### Serializer
- It can save Link, Optimizer, Trainer as npz or hdf5
- Some attributes, e.g., parameters  are serialized automatically, the others are not. For such attributes to be saved, we have to specify by calling Link.add_persistent() method.

### Example: Multi-layer Perceptron on MNIST

This is a hello world example in Deep Learning.



# Note
- For multi-element array, set the initial values for the gradient of the last variable which calls the backward function.
- Gradient is accumulated, not overwritten, so that call f.zerograds() first.

# Reference
- http://docs.chainer.org/en/stable/tutorial/basic.html
- http://docs.chainer.org/en/stable/tutorial/index.html

