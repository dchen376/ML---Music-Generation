from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU,BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
import numpy as np


class Autoencoder:
    """
    Autoencoder represents a deep convolutional autoencoder architecture 
     with mirrored encoder and decoder component.
    """
    def __init__(self,
                 input_shape, #[28, 28, 1] height, width, channels 
                 conv_filters, # Number of filters for each layer [2, 4, 8]
                 conv_kernels, # size of the filter in each layer: kernel [3 5 3]
                 conv_strides, # [1 2 2]
                 latent_space_dim): #the dimensionality for bottomneck = 2
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        self.encoder =  None # a keras model
        self.decoder = None
        self.model = None

        '''private attributes'''
        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None

        self._build() # method; for building and running the class.

    def summary(self):
        self.encoder.summary() #it's a bult-in method in tensorflow.
        self.decoder.summary()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        # self._build_autoencoder()

    """
    Build Decoder.
    """
    def _build_decoder(self):
        decoder_input = self._add_decoder_input() # methods;
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name = "decoder")

    def _add_decoder_input(self):
        return Input(shape = self.latent_space_dim, name = "decoder_input")


    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] ->>8
        dense_layer = Dense(num_neurons, name = "decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order
        # and stop at the first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            # [0, 1, 2] -> [2, 1, 0] --> [1 2] -> [2 1]
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = "same",
            name = f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name = f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name = f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters = 1,
            kernel_size=self.conv_kernels[0],
            strides = self.conv_strides[0],
            padding = "same",
            name = f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name = "sigmoid_layer")(x)
        return output_layer



    """
    Build Encoder.
    """

    def _build_encoder(self):
        encoder_input = self._add_encoder_input() # methods;
        conv_layers = self._add_conv_layers(encoder_input) #;
        bottleneck = self._add_bottleneck(conv_layers) #;
        self.encoder = Model(encoder_input, bottleneck, name = "encoder") #KERAS model


    def _add_encoder_input(self): #use KERAS model ->> layers.Input
        return Input(shape = self.input_shape, name = "encoder_inputs")

    def _add_conv_layers(self, encoder_input):
        """Creates all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Add a convolutional block to a graph of layers, consisting of 
            conv 2d + ReLu + batch normalization.
        """

        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters =  self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = "same",
            name = f"encoder_conv_layer_{layer_number}"
        )  #with KERAS model

        # 'x' is the Input() from KERAS: to apply the conv_layers to the Input().
        x = conv_layer(x) #KERAS model --> Con2D
        x = ReLU(name = f"encoder_relu_{layer_number}")(x) # apply the conv_layer 'x' to reLU
        x = BatchNormalization(name = f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add to bottleneck (Dense layer)"""
        self._shape_before_bottleneck = K.int_shape(x)[1:] # [2, 7, 7, 32]
        x = Flatten()(x) #KERAS
        x = Dense(self.latent_space_dim, name = "encoder_output")(x) #KERAS
        return x


if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape = (28, 28, 1),
        conv_filters = (32, 64, 64, 64),
        conv_kernels = (3, 3, 3, 3),
        conv_strides = (1, 2, 2, 1),
        latent_space_dim=2)
    autoencoder.summary()