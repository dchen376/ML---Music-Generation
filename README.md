# FSDD
The sound dataset was gathered from this git repository -> Free Spoken Digit Dataset (FSDD):
https://github.com/Jakobovski/free-spoken-digit-dataset

# MNIST
The analysis.py is using MNIST (Modified National Institute of Standards and Tehchnology) as a dataset for pre-analysis purpose; dataset of handwritten digits.

# Youtube reference
Youtube tutorials on music generation:
https://youtube.com/playlist?list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&si=53DtJN6I_OKJFAr-



# Steps to follow in this project
step 0 - Understand vanilla autoencoder which consists of both an encoder and a decoder.
  - build an encoder
  - build a decoder
  - combine and make the autoencoder
  - train the autoencoder
  - test the autoencoder with mnist dataset
  - plot the testing results

step 1 - Implement Variational Autoencoder (VAE)
  - modify encoder component (modify the bottleneck -> z = u + sum(epsilon))
  - modify loss function: RMSE + KL (Kullback-Leibler Divergence (closed form))
  - train vae
    
step 2 - Preprocessing Audio Datasets
  - use Free Spoken Digit Dataset (FSDD) (an audio preprocessing library)
  - implement Loader and Padder for file processing
  - implement LogSpectrogramExtractor to preprocess audio files as spectrograms
  - implement MinMaxNormaliser
  - implement the Preprocessing Pipeline
  - implement Saver

step 3 - Training a VAE with speech data in Keras
  - load Free Sound Digits Dataset (FSDD)
  - reshape the data
  - train the VAE

step 4 - Sound Generation with VAE
  - build a SoundGenerator class
  - Implement a generate.py script
  - generate Sound from Spectrograms
