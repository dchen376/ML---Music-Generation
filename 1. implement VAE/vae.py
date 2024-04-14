import os  # operating system dependent functionality.
import pickle  # for serializing and deserializing Python objects: byte stream & object.

from tensorflow.keras import Model  # represents a neural network model.
from tensorflow.keras.layers import (
    Input,  # used to instantiate a Keras tensor, which is a placeholder for the input data.
    Conv2D,  # a class in TensorFlow (specifically in the Keras API) used to create 2D convolutional layers in a neural network.
    ReLU,
    BatchNormalization,
    Flatten,
    Dense,
    Reshape,
    Conv2DTranspose,
    Activation,
Lambda
)

# provides functions and utilities for working with tensors and neural network operations. It serves as an abstraction layer that allows you to write code that is compatible with multiple backends, including TensorFlow
from tensorflow.keras import backend as K  # this is a convention to use 'K'

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import numpy as np
import tensorflow as tf

print(tf.__version__)

# TensorFlow from eager execution mode to graph execution mode. After calling this function, TensorFlow will build a computational graph instead of executing operations immediately.
tf.compat.v1.disable_eager_execution() # is used to disable eager execution mode in TensorFlow 2.x.


def _calculate_reconstruction_loss(y_target, y_predicted):
    error = y_target - y_predicted
    reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
    return reconstruction_loss


def _calculate_kl_loss(model):
    # wrap `_calculate_kl_loss` such that it takes the model as an argument,
    # returns a function which can take arbitrary number of arguments
    # (for compatibility with `metrics` and utility in the loss function)
    # and returns the kl loss
    def _calculate_kl_loss(*args):
        kl_loss = -0.5 * K.sum(1 + model.log_variance - K.square(model.mu) -
                               K.exp(model.log_variance), axis=1)
        return kl_loss
    return _calculate_kl_loss

class VAE:
    """

    VAE represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.

    """

    # constructor
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape  # [28, 28, 1] width, height, # channels (shape or dimensions of the input data that is fed into a neural network layer.
        self.conv_filters = conv_filters  # num filters for each layer [2, 4, 8]
        self.conv_kernels = conv_kernels  # size of the filters (kernels) for each convolutional layer. [3, 3, 3]; first layer has filters of size 3x3.
        self.conv_strides = conv_strides  # [1, 2, 2] num of pixels by which the convolutional kernel moves across the input data at each step.
        self.latent_space_dim = latent_space_dim  # 2
        self.reconstruction_loss_weight = 1000

        self.encoder = None  # several layers of neurons go into the bottleneck.
        self.decoder = None  # several layers out of the bottleneck.
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()  # for building everything.

    # the summary of the encoder.
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()  # since decoder is a keras module.
        self.model.summary()

    # create a compile method
    def compile(self, learning_rate=0.0001):
        # In 2014, Adam (for "Adaptive Moment Estimation")
        # optimizer is a critical component used during the training phase of a model to minimize a predefined loss function and update the model parameters (weights and biases) iteratively.
        optimizer = Adam(learning_rate = learning_rate)  # optimizer: goal of an optimizer is to find the set of parameters that result in the model making accurate predictions on unseen data.

        # compile model:
        self.model.compile(optimizer = optimizer,
                           loss = self._calculate_combined_loss,
                           metrics = [_calculate_reconstruction_loss,
                                      _calculate_kl_loss(self)]
                           )  # native method to keras

    # train model
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,  # input of our training set
                       x_train,  # expected output (the same samples we get as input here).
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True
                       )

    # save model
    def save(self, save_folder="."):  # default to working directory.
        self._create_folder_if_it_doesnt_exist(save_folder)  # save in folder.
        self._save_parameters(save_folder)  # save parameters.
        self._save_weights(save_folder)  # save weights.

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)


    def reconstruct(self, images):
        # encoder to create latent representation
        latent_representations = self.encoder.predict(images) #predict() built in in Keras.
        # pass reconst. to decoder
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod #to indicate that it's a class method.
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(
            *parameters)  # * is unpacking operator: to unpack arguments from an iterable (like a list or tuple).
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder


    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = _calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = _calculate_kl_loss(self)()
        #needs to ADD! reconstruction loss weight (alpha)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    # MSE errors for VAE eqn.
    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis = [1, 2, 3]) #for squared error.
        return reconstruction_loss

    # cal kl loss.
    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = - 0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis = 1) # axis = 1 indicates that the sum operation should be performed along the second axis (axis with index 1) of the tensor.
        return kl_loss

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        # wb: binary write mode.
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)  # parse objects into the file 'f'.

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input  # assign it to when we build the encoder.
        model_output = self.decoder(self.encoder(model_input))  # output of all my autoencoder model.
        self.model = Model(model_input, model_output, name="autoencoder")

    """==================================================================================="""
    """==================================================================================="""
    """2. Build DECODER"""

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()  # implement in later.
        dense_layer = self._add_dense_layer(
            decoder_input)  # dense layer typically consists of fully connected neurons, where each neuron is connected to every neuron in the previous layer.
        reshape_layer = self._add_reshape_layer(
            dense_layer)  # layer reshapes the output of the dense layer into a format suitable for the subsequent convolutional transpose layers.  # pass dense layer n reshape it.
        conv_transpose_layers = self._add_conv_transpose_layers(
            reshape_layer)  # (also known as deconvolutional layers) are used to unsample the feature maps, increasing their spatial resolution/dimensions back. These layers can learn to recover fine-grained details lost during downsampling operations in the encoder part of the network.
        decoder_output = self._add_decoder_output(
            conv_transpose_layers)  # produces the final output of the decoder, which is typically the reconstructed version of the input data.
        self.decoder = Model(decoder_input, decoder_output, name="decoder")


    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")


    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)  # [1, 2, 4] -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)  # keras dense layer
        return dense_layer


    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)


    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the
        # first layer.
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x


    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x


    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,  # [24, 24, 1] #one channel as an output.
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)  # apply this to x; get back to the updated graph of layers.
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer


    """==================================================================================="""
    """==================================================================================="""
    """1. BUILD ENCODER """


    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)  # add to conv layer. # a func defined later.
        # use API of Keras.
        bottleneck = self._add_bottleneck(conv_layers)  # apply to bottleneck (flatten it) # a func defined later.
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")  # an arbitary name: "encoder"



    # next, add the input().
    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")  #


    # 2nd, add convolutional layers.
    # create layers
    def _add_conv_layers(self, encoder_input):
        """Creates all convolutionals blocks in encoder"""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x


    def _add_conv_layer(self, layer_index, x):
        """Adds a convolutional block to a graph of layers,
        consisting of conv 2d + ReLu + batch normalization
        """
        layer_number = layer_index + 1

        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],  # size of filters.
            strides=self.conv_strides[layer_index],
            padding="same",  # output feature map will have the same spatial dimensions as the input.
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)  # get Keras layer and apply to the layer 'x'.
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)  # instansiate Relu() layer, apply it to 'x'.
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)  # apply normalization.
        return x


    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck (Dense layer)."""
        # take data, flatten it, then cast data into dense layer (bottleneck) with Gaussian sampling!

        # store the shapes before flatten it.
        self._shape_before_bottleneck = K.int_shape(x)[1:]  # [7, 7, 32] # getting the shape of the data at 'x' (will be a 4d array): [2, 7, 7, 32] first 2 is the batch size.
        x = Flatten()(x) #  converting multi-dimensional arrays (tensors) into a one-dimensional array.
        # represent the mean vector.
        self.mu = Dense(self.latent_space_dim, name = "mu")(x) #  contains the mean values of a set of data points.
        self.log_variance = Dense(self.latent_space_dim, name = "log_variance")(x)

        def sample_point_from_normal_distribution(args): #args is used as a single tuple parameter containing two elements: mu and log_variance. This approach allows the function to accept a single argument that encapsulates multiple values, making the function call more compact and flexible.
            mu, log_variance = args # means that the first element of the tuple is assigned to the variable mu, and the second element is assigned to the variable log_variance.
            epsilon = K.random_normal(shape = K.shape(self.mu), mean = 0., stddev = 1.)

            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        #lamda layer
        x = Lambda(sample_point_from_normal_distribution,
                   name = "encoder_output")([self.mu, self.log_variance]) #func will define later.
        return x


    """==================================================================================="""
    """==================================================================================="""
    """RUN THE MAIN()"""

if __name__ == "__main__":  # check whether a Python script is being run as the main program or if it is being imported as a module into another script.
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),  # four conv layers. e.g first layer has 32 filters.
        conv_kernels=(3, 3, 3, 3),  # all four layers, each filter using/having size 3x3.
        conv_strides=(1, 2, 2, 1),  # step size.
        latent_space_dim=2
    )
    autoencoder.summary()

    """ DONE WITH THE CODE OF THE AUTO-ENCODER!!"""
