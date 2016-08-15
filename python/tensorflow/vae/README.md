# Variational Auto Encoder (*IN PROGRESS*)

This is an implementation of VAE, Variational Auto Encoder with TensorFlow.

Variational Auto Encoder was originally proposed by Kinga (2013). It is the first work, as far as I know, combining Deep Learning and Variational Inference. There are mainly two neural networks of VAE; Inference Network and Generation Network. Inference Network, also referred to as Interence Network, Recognition Model, or Encoder, inters a latent variable. Generation Network, also referred to as Generative Model, Generative Nework, or Decoder, generates a sample conditoined on a latent variable. Originally, before combining Deep Learning, the variational inference uses approxiamation methods, such as Monte Carlo approximation or Gibbs Sampling, to generate a sample if intermidiate probability distributions (assuming other latent variables) are intractable. However, if we use Gaussian distribution or Multinomial distribution which are used to sampling target latent variables, intermidiate and complicated procedures to generate samples can replaced with Deep Neural Network, and an objective function w.r.t. both parameters of an inference model and generative model can be directly and jointly optimized without intricate and computationally complex sampling techniques. I conjecure it is the main contribution of this work; the fusion of Variational Inference and Deep Learning, and it showed to work properly.

In my work, the followings are the setting of VAE.

- Inference model: Gaussian
- Prior: Gaussian
- Generative model: Bernoulli
- Datataset: permutation invariant (1-d) MNIST (binarized)
- Batch size: 128
- Hidden Units: 260 (=782/3) arbitrarily determined
- Network details: see the [model.py](https://github.com/kzky/languages/blob/master/python/tensorflow/vae/model.py), this might be a bit different from the origin.

The inference model and  prior are both Gaussian, so KL divergence can be computed analytically, and the generative model is Bernoulli. As such, the dataset MNIST should be binaized.


# Resuls of MNIST

Tensorflow  I used is the version 0.9. When you run the main script, you can get the output like the following on the stdout.

```
...
Epoch=163,Elapsed Time=13752.871527[s],Iter=76283,Obj(Test)=-0.790570795536
Epoch=164,Elapsed Time=13837.2553349[s],Iter=76751,Obj(Test)=-0.79049706459
Epoch=165,Elapsed Time=13922.0969388[s],Iter=77219,Obj(Test)=-0.790649235249
Epoch=166,Elapsed Time=14006.484201[s],Iter=77687,Obj(Test)=-0.790925621986
Epoch=167,Elapsed Time=14090.968374[s],Iter=78155,Obj(Test)=-0.791207253933
Epoch=168,Elapsed Time=14175.646683[s],Iter=78623,Obj(Test)=-0.791382491589
Epoch=169,Elapsed Time=14260.0148499[s],Iter=79091,Obj(Test)=-0.790939152241
```

At the very beginning of epoch, the variational lower bound (objective) is very low, but it gradually increases. Learning is very unstable, depending on the numbe of units of MLP and of the latent variables. It gets nan very easily based on my reproduction and learning convergence is very slow in terms of epoch measurement. To prevent the unstability of learning (or becoming nan), one to solve this is to apply *Tanh* to the statictics (*mu* and *log var*) with n-times multiplication. n limits the upper and lower bound for both *mu* and *log var*. That is my experience when trying to reproducing the work, but it can work without using *Tanh* to the statictics if we choose the number of units and of the latent variables.

# References
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes, (Ml), 1–14.
