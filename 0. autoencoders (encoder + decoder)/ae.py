from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU,BatchNormalization, Flatten, Dense
from tensorflow.keras import backend as K


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

    def _build(self):
        self._build_encoder()
        # self._build_decoder()
        # self._build_autoencoder()

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