# Variational Auto Encoder (IN PROGRESS)

This is an implementation of VAE, Variational Auto Encoder with TensorFlow.

Variational Auto Encoder was originally proposed by Kingma (2013). It is the first work, as far as I know, combining Deep Learning and Variational Inference. There are mainly two neural networks of VAE; Inference Network and Generation Network. Inference Network, also referred to as Interence Network, Recognition Model, or Encoder, inters a latent variable. Generation Network, also referred to as Generative Model, Generative Nework, or Decoder, generates a sample conditoined on a latent variable. Originally, before combining Deep Learning, the variational inference uses approxiamation methods, such as Monte Carlo approximation or Gibbs Sampling, to generate a sample if the posterior probability distributions are intractable. However, if we use Gaussian distribution or Multinomial distribution which are used to sampling latent variables, intermidiate and complicated procedures to generate samples can replaced with Deep Neural Network, and an objective function w.r.t. both parameters of an inference model and generative model can be directly and jointly optimized without intricate and computationally complex sampling techniques. I conjecure it is the main contribution of this work; the fusion of Variational Inference and Deep Learning, and it showed to work properly.

In my work, the followings are the settings of VAE.

- Inference model: Gaussian
- Prior: Gaussian
- Generative model: Bernoulli
- Datataset: permutation invariant (1-d) MNIST (binarized)
- Batch size: 100
- Hidden units: 500
- Latent variables: 20
- Network details: see the [model.py](https://github.com/kzky/languages/blob/master/python/tensorflow/vae/model.py), this might be a bit different from the origin.
- More details can be found in [main.py](https://github.com/kzky/languages/blob/master/python/tensorflow/vae/main.py)

The inference model and  prior are both Gaussian, so KL divergence can be computed analytically, and the generative model is Bernoulli. As such, the dataset MNIST should be binaized.

# Results of MNIST

```sh
...
Epoch=34,Elapsed Time=54.4531319141[s],Iter=20399,Obj(Test)=-118.486526489
Epoch=35,Elapsed Time=56.0464878082[s],Iter=20999,Obj(Test)=-117.570510864
Epoch=36,Elapsed Time=57.6403009892[s],Iter=21599,Obj(Test)=-117.194610596
Epoch=37,Elapsed Time=59.2312989235[s],Iter=22199,Obj(Test)=-116.218093872
Epoch=38,Elapsed Time=60.8325719833[s],Iter=22799,Obj(Test)=-116.471473694
Epoch=39,Elapsed Time=62.4199149609[s],Iter=23399,Obj(Test)=-116.015098572
Epoch=40,Elapsed Time=64.0109570026[s],Iter=23999,Obj(Test)=-114.883216858
Epoch=41,Elapsed Time=65.6005599499[s],Iter=24599,Obj(Test)=-112.517486572
...
```

Tensorflow  I used is the version 0.9. When you run the main script, you can get the output like the following on the stdout.

Learning is very unstable, it gets nan very easily based on my reproduction, depending on the initial value of weights. If the initial value of weights is small enouph, e.g., 0.5 x random gaussina.

# TODO
- Visualize the reconstruction.

# References
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes, (Ml), 1–14.
