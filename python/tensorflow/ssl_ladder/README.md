# Semi-supervised Ladder Network Implementation

This is an implementation of Semi-supervised Ladder Network Implementation by Tesnorflow.

Semi-supervised Ladder Network was proposed by Rasmus et al. It can be used for semi-supervised learning (SSL) and composes of the three network; the corrupted encoder, the dencoder, and the clean encoder. The correupted encoder adds a noise to a sample when encoding, and the decoder tries to decode the corrupted input from the corresponding layer in the encoder with the denoised estimate from the above layer. This can be seen as the denoising autoencoder (dAE) with lateral connections; however, there is the clean encoder which features are used for denoising cost (reconstruction cost) to the denoised estimates, and it shares trainable parameters with those of the corrupted encoder. In addition to the denoising cost in the objective function, there are also a classification loss, so that Semi-supervised Ladder Network can be used for SSL. Note that the classification loss is between the target value and the noisy prediction, so the noise also regularizes the supervised learning. 

When reconstructing the noisy inputs, the key to reconstrcut is such a way to fomulate the probablity distribution of the reconstructed sample same as that of the original sample before adding a noise and add the reconstruction cost between samples nomarlized in the same way, i.e., use the same batch mean and variance. In order to use the same distributions, we have to know the mean and variance of the reconstructed sample so make these functions of the reconstructed samples of the above layer. Note these functions have trainable parameters; a neural net. For reconstruction error, use the batch mean and variance of the clean samples. Using the variance-scaled least-square loss between the sample and the denoised sample results in this reconstruction error.

# Result
When the proposed architecture (*Ladder, full*) in [1] is used, it achieved around 98% with 100 labeled samples only. More performance may be able to be achieved with more hyper-parameter exploration.

# Notice
- For the reconstruction loss at the input layer, I used *z_hat* as *z_BN* because there are no clear explanation in the paper for how to deal with the input layer.
- Choice of values of gaussian noise and initialization for per-element trainable parameters of the denoising function are critial for the fast learning and high performance, it is often hard.

# References
1. Rasmus, A., Valpola, H., Honkala, M., Berglund, M., & Raiko, T. (2015). Semi-Supervised Learning with Ladder Networks. Neural and Evolutionary Computing; Learning; Machine Learning.
2. Pezeshki, M., Fan, L., Brakel, P., Courville, A., & Bengio, Y. (2016). Deconstructing the Ladder Network Architecture (icml). Icml, 48, 1–15.

