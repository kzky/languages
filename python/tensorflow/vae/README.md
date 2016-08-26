# Variational Auto Encoder

This is an implementation of VAE, Variational Auto Encoder with TensorFlow.

Variational Auto Encoder was originally proposed by Kingma (2013). It is the first work, as far as I know, combining Deep Learning and Variational Inference. There are mainly two neural networks of VAE; Inference Network and Generation Network. Inference Network, also referred to as Interence Network, Recognition Model, or Encoder, inters a latent variable. Generation Network, also referred to as Generative Model, Generative Nework, or Decoder, generates a sample conditoined on a latent variable. Originally, before combining Deep Learning, the variational inference uses approxiamation methods, such as Monte Carlo approximation or Gibbs Sampling, to generate a sample if the posterior probability distributions are intractable. However, if we use Gaussian distribution or Multinomial distribution which are used to sampling latent variables, intermidiate and complicated procedures to generate samples can replaced with Deep Neural Network, and an objective function w.r.t. both parameters of an inference model and generative model can be directly and jointly optimized without intricate and computationally complex sampling techniques. I conjecure it is the main contribution of this work; the fusion of Variational Inference and Deep Learning, and it showed to work properly.

In my work, the followings are the settings of VAE.

- Inference model: Gaussian
- Prior: Gaussian
- Generative model: Bernoulli
- Datataset: permutation invariant (1-d) MNIST (binarized)
- Batch size: 128
- Hidden units: 300
- Latent variables: 100
- Network details: see the [model.py](https://github.com/kzky/languages/blob/master/python/tensorflow/vae/model.py), this might be a bit different from the origin.
- More details can be found in [main.py](https://github.com/kzky/languages/blob/master/python/tensorflow/vae/main.py)

The inference model and  prior are both Gaussian, so KL divergence can be computed analytically, and the generative model is Bernoulli. As such, the dataset MNIST should be binaized.


# Results of MNIST

```sh
...
Epoch=203,Elapsed Time=18811.7662501[s],Iter=95003,Obj(Test)=-358.427459717
Epoch=204,Elapsed Time=18905.1899111[s],Iter=95471,Obj(Test)=-357.365539551
Epoch=205,Elapsed Time=18995.8811719[s],Iter=95939,Obj(Test)=-358.109130859
Epoch=206,Elapsed Time=19089.237061[s],Iter=96407,Obj(Test)=-359.817718506
Epoch=207,Elapsed Time=19179.869525[s],Iter=96875,Obj(Test)=-357.824981689
Epoch=208,Elapsed Time=19271.271457[s],Iter=97343,Obj(Test)=-357.644866943
Epoch=209,Elapsed Time=19365.1253309[s],Iter=97811,Obj(Test)=-361.130554199
Epoch=210,Elapsed Time=19457.0572[s],Iter=98279,Obj(Test)=-359.074066162
Epoch=211,Elapsed Time=19549.4762731[s],Iter=98747,Obj(Test)=-355.625030518
Epoch=212,Elapsed Time=19642.673737[s],Iter=99215,Obj(Test)=-362.736236572
Epoch=213,Elapsed Time=19734.7681401[s],Iter=99683,Obj(Test)=-362.651245117
```

Tensorflow  I used is the version 0.9. When you run the main script, you can get the output like the following on the stdout.

At the very beginning of epoch, the variational lower bound (objective) is very low, but it gradually increases. Learning is very unstable, depending on the number of units of MLP and of the latent variables. It gets nan very easily based on my reproduction, and the convergence is very slow in terms of epoch measurement. To prevent the unstability of learning (or becoming nan), one way to solve this is using *clipping* to the statictics (*mu* and *log var*). That is my experience when trying to reproducing the work, but it can work without using *clipping* to the statictics if we choose the number of units and of the latent variables carefully.

The remaining of this work is to visualize the reconstruction.

# References
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes, (Ml), 1–14.
