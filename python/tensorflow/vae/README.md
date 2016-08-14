# Variational Auto Encoder

This is an implementation of VAE, Variational Auto Encoder with TensorFlow.

Variational Auto Encoder was originally proposed by Kinga (2013). It is a first work, as far as I know, combining Deep Learning and Variational Inference.

There are mainly two neural networks; Inference Network and Generation Network. Inference Network, also referred to as Interence Network or Recognition Model, inters a latent variable $\\math{y}$ . Generation Network, also referred to as Generative Model or Generative Nework, generates a sample $\\mathbf{x}$ conditoined on a latent variable $\\mathbf{y}$. Originally, before combining Deep Learning, the variational inference uses approxiamation methods, such as Monte Carlo approximation or Gibbs Sampling, to generate a sample if intermidiate probability distributions are intractable. However, if we use Gaussian distribution or Multinomial distribution which are used to sampling, intermidiate and complicated procedures to generate sample can replaced with Deep Neural Network and an objective function can be directly optimized without intricate and computationally complex sampling techniques. I conjecure it is the main contribution of this work; the fusion of Variational Inference and Deep Learning, and it showed to work properly.


# References
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes, (Ml), 1–14. Retrieved from http://arxiv.org/abs/1312.6114
