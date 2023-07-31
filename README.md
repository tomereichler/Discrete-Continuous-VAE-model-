# Discrete-Continuous-VAE-model-

In this work,there is a VAE model with latent spaces that is both continuous and discrete
- Model: There are many ways to implement this model. here, we concatenated the 2 latent spaces from the discrete and continuous encoders and decoded them 
- Dataset: we use the MNIST dataset. Differently, this time we randomly
colored each sample with a different color. We implemented this using a Dataset
class that is wrapping the MNIST dataset.
- Outputs visualization: We generated images from the model and visualize its latent
spaces. 
