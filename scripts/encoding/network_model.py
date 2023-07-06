import numpy as np
from tensorflow import keras

from scripts.encoding.constants import ActivationFunctions
from scripts.encoding.symb_backward_bounds_calc import LinearEquation


class NeuralNetworkLayer(object):
    """
    A class representing a layer of a neural network.
    This class is not associated with any activation function.

    depth is the number of the layer in a network starting from 1
    """
    def __init__(self, depth):
        self.depth = depth

    def get_depth(self):
        return self.depth


class FlattenLayer(NeuralNetworkLayer):
    """
    A class representing a Flatten layer of a convolutional neural network.

    input_shape is the shape of the feature maps that are being flattened

    output_size is the number of nodes after flatenning
    (should be equal to prod(*output_size)

    NOT created anymore by the parser.
    """
    def __init__(self, input_shape, output_size, depth):
        self.input_shape = input_shape
        self.output_size = output_size

        super().__init__(depth)

    def clone(self):
        return FlattenLayer(self.input_shape, self.output_size, self.depth)


class TransformationLayer(NeuralNetworkLayer):
    """
    A class representing a layer of a neural network that performs
    a linear transformation. Therefore, it stores weights and bias.
    This class is not associated with any activation function.
    """
    def __init__(self, weights, bias, depth):
        self.weights = weights
        self.bias = bias

        super().__init__(depth)

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_dot_product(self, gurobi_variables):
        """
        Method for computing the dot product of the (input) gurobi variables
        with the linear transformation represented by the layer.

        Used for computing a Gurobi encoding of the network.

        This implementation is valid for simple fully-connected layers.
        Convolutional layer has to override it.
        """

        dot_product = [
            self.weights[i].dot(gurobi_variables) + self.bias[i]
            for i in range(self.weights.shape[0])
        ]
        return dot_product

    def compute_output_equation(self, input_lower_eq, input_upper_eq):
        """
        Compute new lower and upper bound equations,
        from input lower and upper equations and
        a linear transformation in the form of weights and bias as they are in the network layer.
        """

        weights_plus = np.maximum(self.weights, np.zeros(self.weights.shape))
        weights_minus = np.minimum(self.weights, np.zeros(self.weights.shape))

        # get coefficients for the input bound equations
        output_lower_matrix = weights_plus.dot(input_lower_eq.get_matrix()) + \
                              weights_minus.dot(input_upper_eq.get_matrix())
        output_upper_matrix = weights_plus.dot(input_upper_eq.get_matrix()) + \
                              weights_minus.dot(input_lower_eq.get_matrix())

        # get constants for the input bound equations
        output_lower_offset = weights_plus.dot(input_lower_eq.get_offset()) + \
                              weights_minus.dot(input_upper_eq.get_offset()) + \
                              self.bias
        output_upper_offset = weights_plus.dot(input_upper_eq.get_offset()) + \
                              weights_minus.dot(input_lower_eq.get_offset()) + \
                              self.bias

        return LinearEquation(output_lower_matrix, output_lower_offset), \
               LinearEquation(output_upper_matrix, output_upper_offset)

    def as_linear_equation(self):
        return LinearEquation(self.weights, self.bias)


class ReluLayer(TransformationLayer):
    """
    A class representing a linear transformation layer of a neural network
    with ReLU activation function.
    """
    def __init__(self, weights, bias, depth):
        super().__init__(weights, bias, depth)


class DenseReluLayer(ReluLayer):
    """
    A class representing a fully connected layer of a neural network with ReLU activation function.
    """
    def __init__(self, weights, bias, depth):
        self.input_size = weights.shape[1]
        self.output_size = weights.shape[0]

        super().__init__(weights, bias, depth)

    def clone(self):
        return DenseReluLayer(self.weights, self.bias, self.depth)


class LinearLayer(TransformationLayer):
    """
    A class representing a fully connected layer of a neural network with Linear activation function.
    """
    def __init__(self, weights, bias, depth):
        self.input_size = weights.shape[1]
        self.output_size = weights.shape[0]

        super().__init__(weights, bias, depth)

    def clone(self):
        return LinearLayer(self.weights, self.bias, self.depth)


class KerasParser:

    @staticmethod
    def parse_file(nn_filename):
        nmodel = keras.models.load_model(nn_filename)
        return KerasParser.parse_model(nmodel)

    @staticmethod
    def parse_model(nmodel):
        layers = []
        keras_layer_counter = 0
        # network layers are counted from 1
        layer_counter = 1

        # Initialise the layers on the network in our format
        while keras_layer_counter < len(nmodel.layers):

            keras_layer = nmodel.layers[keras_layer_counter]

            if isinstance(keras_layer, keras.layers.Dense):
                keras_layer_counter, layer = \
                    KerasParser._parse_keras_dense_layer(nmodel, keras_layer_counter, layer_counter)

                layers.append(layer)
                keras_layer_counter += 1
                layer_counter += 1
            elif isinstance(keras_layer, keras.layers.Flatten):
                # # Flatten layer is not added anymore to the network model
                # keras_layer_counter, layer = \
                #     KerasParser._parse_keras_flatten_layer(nmodel, keras_layer_counter, layer_counter)
                #
                # layers.append(layer)
                keras_layer_counter += 1
                # layer_counter += 1
            elif isinstance(keras_layer, keras.layers.Reshape):
                assert len(keras_layer.output_shape) == 2, "Only accepting Reshape layers that act as Flatten layers"
                keras_layer_counter += 1
            elif isinstance(keras_layer, keras.layers.InputLayer):
                keras_layer_counter += 1
            else:
                raise Exception("Unsupported network layer: {}.\n "
                                "Expected a Dense, Flatten, Reshape, or InputLayer layer.".format(keras_layer))

        n_layers = layer_counter - 1
        input_shape = nmodel.input_shape[1:]
        if len(input_shape) == 1:
            input_shape = input_shape[0]

        return layers, n_layers, input_shape

    @staticmethod
    def _parse_keras_dense_layer(nmodel, keras_layer_counter, layer_counter):
        """
        Parse a Dense layer.
        Since our layers always include an activation function,
        while keras layers might have the actual activation in a separate layer,
        we might also have a look at the next layer. In the latter case we return an updated index i

        :param nmodel: keras model
        :param keras_layer_counter: the index of the current (dense) layer in nmodel
        :param layer_counter: the depth of the current layer in our internal representation
        :return:
            i+1 if the next layer is an activation layer, i otherwise,
            the layer corresponding to the dense layer
        """
        layer = nmodel.layers[keras_layer_counter]

        weights = layer.get_weights()[0].T
        bias = layer.get_weights()[1]

        # Variable storing the layer's activation function
        # Currently supporting only ReLU and Linear.
        activation = ActivationFunctions.UNKNOWN

        # detect the activation function
        if layer.activation == keras.activations.relu:
            activation = ActivationFunctions.RELU
        elif layer.activation == keras.activations.softmax:
            activation = ActivationFunctions.LINEAR
        elif layer.activation == keras.activations.linear:
            # Below we check for the relevant activation function
            # that could be encoded in the next keras layer
            keras_layer_counter, activation = KerasParser.check_activation_in_next_layer(nmodel, keras_layer_counter)

        # return depending on the value of activation
        if activation == ActivationFunctions.RELU:
            return keras_layer_counter, DenseReluLayer(weights, bias, layer_counter)
        elif activation == ActivationFunctions.LINEAR:
            return keras_layer_counter, LinearLayer(weights, bias, layer_counter)
        else:
            raise Exception("Unsupported activation function", layer.activation)


    @staticmethod
    def check_activation_in_next_layer(nmodel, keras_layer_counter):
        """
        Method for checking if the activation function is stored in the next keras layer.
        Should only be called when checking a layer with linear activation.
        If it finds that the next layer is an Activation layer,
        updates the index of the current keras layer
        :param keras_layer_counter: the index of the current (non Activation) layer
        :param nmodel: keras model
        :return:
        """

        assert nmodel.layers[keras_layer_counter].activation == keras.activations.linear
        activation = ActivationFunctions.LINEAR

        if keras_layer_counter + 1 < len(nmodel.layers):
            if isinstance(nmodel.layers[keras_layer_counter + 1], keras.layers.Activation):
                keras_layer_counter += 1
                layer = nmodel.layers[keras_layer_counter]

                if layer.activation == keras.activations.relu:
                    activation = ActivationFunctions.RELU
                elif layer.activation == keras.activations.softmax:
                    # We can also accept softmax as we can compute argmax
                    # using MILP constraints
                    activation = ActivationFunctions.LINEAR
                elif layer.activation != keras.activations.linear:
                    raise Exception("Unsupported activation function", layer.activation, "in layer", keras_layer_counter)

        return keras_layer_counter, activation


class NetworkModel(object):
    """
    A class for an internal representation of a feed-forward neural network.
    Consists of an array of layers, each storing the relevant parameters
    such as weights, bias.

    After parsing, for a network consisting of
    an input layer 0, hidden layers 1,...,k-1 and an output layer k:
      -- n_layers equals k,
      -- layers correspond layers 1,...,k
      -- input_shape corresponds to the shape of the input layer, assumed to be a tuple

    """
    def __init__(self, layers=[], n_layers=0, input_shape=(0,)):
        self.layers = layers
        self.n_layers = n_layers
        # assumed to be a tuple
        self.input_shape = input_shape
        import numpy as np
        self.input_size = np.prod(input_shape)

        self.parsed_from_file = False

    def clone(self):
        new_model = NetworkModel()
        for layer in self.layers:
            new_model.layers.append(layer.clone())

    def get_layer(self, layer_n):
        assert 1 <= layer_n <= self.n_layers
        return self.layers[layer_n-1]

    def parse(self, nn_filename):
        if nn_filename.endswith(".h5"):

            layers, n_layers, input_shape = KerasParser.parse_file(nn_filename)

            self.parsed_from_file = True
        else:
            raise Exception("Unsupported network model file format", nn_filename)

        self.layers = layers
        self.n_layers = n_layers
        self.input_shape = input_shape
        import numpy as np
        self.input_size = np.prod(input_shape)

    def add_layer(self, layer):
        """
        A method for manually constructing the network model.
        Can only be used if the model hasn't been already parsed from a file.
        """
        assert not self.parsed_from_file

        self.layers.append(layer)
        self.n_layers += 1

    def set_input_size(self):
        """
        A method for manually constructing the network model
        Can only be used if the model hasn't been already parsed from a file.
        Should only be called after adding at least one layer.
        """
        assert not self.parsed_from_file
        assert len(self.layers) > 0

        self.input_shape = self.layers[0].input_shape

