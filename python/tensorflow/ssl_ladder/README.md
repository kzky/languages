# Semi-supervised Ladder Network Implementation (IN PROGRESS)

This is an implementation of Semi-supervised Ladder Network Implementation by Tesnorflow.

Semi-supervised Ladder Network was proposed by Rasmus et al. It can be used for semi-supervised learning (SSL) and composes of the three network; a corrupted encoder, a dencoder, and a clean encoder. The correupted encoder adds a noise to a sample when encoding a sample, and the decoder tries to decode the corrupted input as much as possible. This can be seen as the denoising autoencoder (dAE); however, there is a clean encoder which is used for denoising cost between corrupted samples and denoised ones, and it shares trainable parameters with those of the corrupted encoder. In addition to the denoising cost in the objective function, there are also a classification loss, so that Semi-supervised Ladder Network can be used for SSL. Note that the classification loss is between the target value and the noisy prediction, so the noise also regularizes the supervised learning. 

When reconstructing the noisy inputs, the key to reconstrcut is such a way that fomulate the probablity distribution of the reconstructed sample same as that of the original sample before adding a noise and add the reconstruction cost between samples nomarlized in the same way, i.e., use the same batch mean and variance. In order to use the same distributions, we have to know the mean and variance of the reconstructed sample so make these functions of the reconstructed samples of the above layer. Note these functions have trainable parameters; a neural net. For reconstruction error, use the batch mean and variance of the before-corrupted samples. Using the variance-scaled least-square loss between the sample and the denoised sample results in this reconstruction error.

#TODO
- Have to address reconstruction for the input.


# References
- Rasmus, A., Valpola, H., Honkala, M., Berglund, M., & Raiko, T. (2015). Semi-Supervised Learning with Ladder Networks. Neural and Evolutionary Computing; Learning; Machine Learning.
- Pezeshki, M., Fan, L., Brakel, P., Courville, A., & Bengio, Y. (2016). Deconstructing the Ladder Network Architecture (icml). Icml, 48, 1–15.

