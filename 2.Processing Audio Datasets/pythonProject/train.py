from vae import VAE

from tensorflow.keras.datasets import mnist #import mnist

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
#20 -> 100 epochs
EPOCHS = 100


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #apply normalization
    x_train = x_train.astype("float32") / 255 # to convert the data type of the elements in x_train to 32-bit floating-point numbers.
    #(1,) add another dimension, which is the channel dimension!
    x_train = x_train.reshape(x_train.shape + (1,)) # change the shape of an array without changing its data.

    x_test = x_test.astype("float32")/255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),  # four conv layers. e.g first layer has 32 filters.
        conv_kernels=(3, 3, 3, 3),  # all four layers, each filter using/having size 3x3.
        conv_strides=(1, 2, 2, 1),  # step size.
        latent_space_dim = 2
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
    x_train, _, _, _ = load_mnist() #IMPLEMENT func above

    # autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS) #implet func above.

    #only run 500  / 10000 samples
    autoencoder = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS) #implet func above.

    # save the autoencoder
    autoencoder.save("model")

    #load the model
    # autoencoder2 = VAE.load("model")
    #
    # autoencoder2.summary()