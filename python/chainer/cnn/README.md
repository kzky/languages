# Convolutional Neural Network with Batch

Implementation of Convolution Neural Network in Chainer is very eas just deriving from [MNIST Example](http://docs.chainer.org/en/stable/tutorial/basic.html#example-multi-layer-perceptron-on-mnist). Even if using BatchNormalization, we just add *L.BatchNorm* either after Conv/Affine Layer or before activation.

One thing which should be mentioned here is if Linear follows Convolution, we generally do not know the spatial shape of feature maps because it depends on the spatial size of input, paddings and strides of convolution kernels. However, we can use *None* to the input dimension at the fist *L.linear*.

# References
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
- http://docs.chainer.org/en/stable/reference/links.html#chainer.links.BatchNormalization
- http://docs.chainer.org/en/stable/reference/links.html#chainer.links.Linear
