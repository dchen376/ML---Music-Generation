from vae import VAE
import os
import numpy as np

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
#20 -> 100 epochs
EPOCHS = 150

SPECTROGRAMS_PATH = r"D:\personal projects\Music Generation\samples\SPECTROGRAMS_SAVE_DIR"


def load_fsdd(spectrograms_path):
    x_train = [] #an empty list to be filled with spectrograms.
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # now, had the spectrogram.
            # (n_bins which from stft, n_frames, 1) #1 is the extra dimension.
            x_train.append(spectrogram) # stored in x_train list.
    x_train = np.array(x_train) # cast to numpy array.
    x_train = x_train[..., np.newaxis] # np.newaxis is used to increase the dimensionality of an array by one unit. # -> (3000 # of data we have in dataset, 256 num bins, 64 frames, 1) # treat spectrograms as grey scale images.
    return x_train






def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),  # four conv layers. e.g first layer has 32 filters.
        conv_kernels=(3, 3, 3, 3, 3),  # all four layers, each filter using/having size 3x3.
        conv_strides=(2, 2, 2, 2, (2, 1)),  # step size.
        latent_space_dim = 128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


"""==================================================================================="""
"""==================================================================================="""
"""RUN THE MAIN()"""
if __name__ == "__main__":
    #MNIST dataset
    x_train = load_fsdd(SPECTROGRAMS_PATH) #IMPLEMENT func above

    # autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS) #implet func above.

    #only run 500  / 10000 samples
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS) #implet func above.

    # save the autoencoder
    autoencoder.save("model")

    #load the model
    # autoencoder2 = VAE.load("model")
    #
    # autoencoder2.summary()