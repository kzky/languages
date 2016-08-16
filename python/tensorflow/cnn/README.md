# Convolutional Neural Nework (IN PROGRESS)

This is an example of Convolutional Neural Network (CNN) with Batch Normalization using MNIST (toy) dataset. Network is very simple 2-layer convolution with max pooling and 2-layer affine and with batch normalization.

Settings are as follows.

- Network: CNN
- Activation: ReLu
- BatchNorm with running mean
- Detail: [models.py](https://github.com/kzky/languages/blob/master/python/tensorflow/cnn/models.py)

## BatchNormalization with running mean, or exponential moving average

[Stack Overflow](http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177?noredirect=1#comment55758348\_33950177) answered by "dominik andreas" explains breifly how to compute and use running mean. Tricks to make this work are using *tf.control_dpendencies* with *moving average op* as input and defining *tf.identity(mean), tf.identify(var)* inside its context, and using *tf.cond* to return either *tf.identity(mean), tf.identify(var)* if in training phase or the moving average of batch\_mean and batch\_var if in test base on the bool of tf.placeholder. These combination makes sure that the moving average is computed before computing *tf.identity* so we can retain the moving average in training phase.

# Results of CNN
[Now computing]

# Referneces
- https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html
- https://www.tensorflow.org/versions/r0.10/api\_docs/python/nn.html#batch\_normalization
- https://www.tensorflow.org/versions/r0.10/api\_docs/python/nn.html#moments
- http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177?noredirect=1#comment55758348\_33950177
- https://www.tensorflow.org/versions/r0.10/api\_docs/python/train.html#ExponentialMovingAverage
- https://www.tensorflow.org/versions/r0.10/api\_docs/python/framework.html#control_dependencies
- https://github.com/tensorflow/tensorflow/blob/b826b79718e3e93148c3545e7aa3f90891744cc0/tensorflow/contrib/layers/python/layers/layers.py#L100
