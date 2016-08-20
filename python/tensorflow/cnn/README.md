# Convolutional Neural Nework

This is an example of Convolutional Neural Network (CNN) with Batch Normalization using MNIST (toy) dataset. Network is very simple 2-layer convolution with max pooling and 2-layer affine and with batch normalization.

Settings are as follows.

- Network: CNN
- Activation: ReLu
- BatchNorm with running mean
- Batch size: 128 * 2
- Detail: [models.py](https://github.com/kzky/languages/blob/master/python/tensorflow/cnn/models.py)

## BatchNormalization with running mean, or exponential moving average

[Stack Overflow](http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177?noredirect=1#comment55758348\_33950177) answered by "dominik andreas" explains breifly how to compute and use running mean. Tricks to make this work are using *tf.control_dpendencies* with *moving average op* as input and defining *tf.identity(mean), tf.identify(var)* inside its context, and using *tf.cond* to return either *tf.identity(mean), tf.identify(var)* if in training phase or the moving average of batch\_mean and batch\_var if in test base on the bool of tf.placeholder. These combination makes sure that the moving average is computed before computing *tf.identity* so we can retain the moving average in training phase.

# Results of CNN

With batch size being 256, 

```txt
...
Epoch=1,Elapsed Time=68.3939840794[s],Iter=233,Loss=113.026544681,Acc=65.5048076923
Epoch=2,Elapsed Time=134.06575799[s],Iter=467,Loss=100.058510517,Acc=71.4443108974
Epoch=3,Elapsed Time=199.20528698[s],Iter=701,Loss=92.4780416183,Acc=75.530849359
Epoch=4,Elapsed Time=265.678523064[s],Iter=935,Loss=90.4514592427,Acc=77.4739583333
Epoch=5,Elapsed Time=332.038901091[s],Iter=1169,Loss=82.2020487908,Acc=81.6806891026
Epoch=6,Elapsed Time=397.261203051[s],Iter=1403,Loss=76.7068933218,Acc=85.2864583333
Epoch=7,Elapsed Time=463.22240901[s],Iter=1637,Loss=72.9553088164,Acc=87.4098557692
Epoch=8,Elapsed Time=529.231127977[s],Iter=1871,Loss=65.3091582732,Acc=91.1858974359
Epoch=9,Elapsed Time=594.179444075[s],Iter=2105,Loss=59.4170658252,Acc=93.6298076923
Epoch=10,Elapsed Time=660.808887959[s],Iter=2339,Loss=56.2977304061,Acc=94.4611378205
Epoch=11,Elapsed Time=726.363693953[s],Iter=2573,Loss=53.785049304,Acc=94.7616185897
Epoch=12,Elapsed Time=792.8128829[s],Iter=2807,Loss=50.0790264362,Acc=95.843349359
Epoch=13,Elapsed Time=859.433815002[s],Iter=3041,Loss=48.4808231011,Acc=95.9134615385
Epoch=14,Elapsed Time=925.235138893[s],Iter=3275,Loss=46.2508573746,Acc=96.3141025641
...

```

As memtioned in somewhere, TF is very slow. To speed up computation in time, increase a batch size, but it also reduces the speed of convergence in terms of epoch measurement.

# Referneces
- https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html
- https://www.tensorflow.org/versions/r0.10/api\_docs/python/nn.html#batch\_normalization
- https://www.tensorflow.org/versions/r0.10/api\_docs/python/nn.html#moments
- http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177?noredirect=1#comment55758348\_33950177
- https://www.tensorflow.org/versions/r0.10/api\_docs/python/train.html#ExponentialMovingAverage
- https://www.tensorflow.org/versions/r0.10/api\_docs/python/framework.html#control_dependencies
- https://github.com/tensorflow/tensorflow/blob/b826b79718e3e93148c3545e7aa3f90891744cc0/tensorflow/contrib/layers/python/layers/layers.py#L100
