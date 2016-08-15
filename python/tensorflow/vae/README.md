# Variational Auto Encoder

This is an implementation of VAE, Variational Auto Encoder with TensorFlow.

Variational Auto Encoder was originally proposed by Kinga (2013). It is the first work, as far as I know, combining Deep Learning and Variational Inference. There are mainly two neural networks of VAE; Inference Network and Generation Network. Inference Network, also referred to as Interence Network, Recognition Model, or Encoder, inters a latent variable. Generation Network, also referred to as Generative Model, Generative Nework, or Decoder, generates a sample conditoined on a latent variable. Originally, before combining Deep Learning, the variational inference uses approxiamation methods, such as Monte Carlo approximation or Gibbs Sampling, to generate a sample if intermidiate probability distributions (assuming other latent variables) are intractable. However, if we use Gaussian distribution or Multinomial distribution which are used to sampling target latent variables, intermidiate and complicated procedures to generate samples can replaced with Deep Neural Network, and an objective function w.r.t. both parameters of an inference model and generative model can be directly and jointly optimized without intricate and computationally complex sampling techniques. I conjecure it is the main contribution of this work; the fusion of Variational Inference and Deep Learning, and it showed to work properly.

In my work (IN PROGRESS), the followings are the setting of VAE.

- Inference model: Gaussian
- Prior: Gaussian
- Generative model: Bernoulli
- Datataset: permutation invariant MNIST (binarized)
- Batch size: 128
- Hidden Units: 260 (=782/3)
- Network details: see the [model.py](https://github.com/kzky/languages/blob/master/python/tensorflow/vae/model.py), this might be a bit different from the origin.

The inference model and  prior are both Gaussian, so KL divergence can be computed analytically, and the generative model is Bernoulli. As such, the dataset MNIST should be binaized.

# References
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes, (Ml), 1–14.
