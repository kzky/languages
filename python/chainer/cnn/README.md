# Convolutional Neural Network with Batch

Implementation of Convolution Neural Network in Chainer is very easy just deriving from [MNIST Example](http://docs.chainer.org/en/stable/tutorial/basic.html#example-multi-layer-perceptron-on-mnist). Even if using BatchNormalization, we just add *L.BatchNorm* either after Conv/Affine Layer or before activation.

Be careful about setting *device* in *trainer/evaluator* in addtion to *model.to\_gpu(device)*

# Results

It can work as complying with the intention.

```txt
epoch       main/accuracy  validation/main/accuracy                                                                                   [5/1936]
1           0.913046       0.964399                  
2           0.967817       0.975079                  
3           0.97463        0.978441                  
4           0.978849       0.980914                  
5           0.981593       0.983683                  
6           0.983759       0.983979                  
7           0.985241       0.985562                  
8           0.986478       0.985957                  
9           0.987523       0.986254                  
10          0.98839        0.98665                   
11          0.989172       0.986946                  
12          0.989483       0.986946                  
13          0.990705       0.987441                  
14          0.991221       0.987737                  
15          0.991804       0.988133                  
16          0.992538       0.988331                  
17          0.992554       0.988034                  
18          0.992737       0.988133                  
19          0.993454       0.988528                  
20          0.993439       0.988825     
```

# References
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
- http://docs.chainer.org/en/stable/reference/links.html#chainer.links.BatchNormalization
- http://docs.chainer.org/en/stable/reference/links.html#chainer.links.Linear
