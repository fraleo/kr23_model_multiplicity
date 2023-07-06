import numpy as np

from scripts.encoding.constants import ReluNodeEncoding, ReluNodeStatus
from scripts.encoding.symb_backward_bounds_calc import IntervalBounds


class NodeVariableLayer(object):
    """
    A data structure containing a description of each node in a layer of a neural network
    such as

    * the concrete interval preactivation bounds
    * the concrete interval bounds
    * the symbolic bounds (lower and upper bound equation)

    * how the node is to be encoded
      -- full MILP encoding
      -- over-approximation (linear or some other relaxation, e.g., quadratic or semidefinite)
      -- under-approximation (fixing nodes to an active on inactive state)

    * the status of a ReLU node (active, inactive, unstable)

    * the over-approximation area of a ReLU node
    * the over-approximation areas sorted

    """
    def __init__(self, size, index, is_relu_activated):
        self.index = index
        self.size = size
        self.preactivation_bounds = IntervalBounds(np.array([float("-inf") for _ in range(size)]),
                                                   np.array([float("inf") for _ in range(size)]))
        self.bounds = IntervalBounds(np.array([float("-inf") for _ in range(size)]),
                                     np.array([float("inf") for _ in range(size)]))
        self.symbolic_bounds = None

        """
        By default, we set the encoding to None
        """
        self.encodings = [ReluNodeEncoding.UNDEFINED for _ in range(size)]

        """
        Initially, we set the status of the ReLU nodes to be UNSTABLE.
        If the layer is not ReLU-activated, the status object is set to None.
        """
        self.status = [ReluNodeStatus.UNSTABLE for _ in range(size)] if is_relu_activated else None
        self.stable_number = -1

    def set_preactivation_symbolic_bounds(self, new_symbolic_bounds):
        """
        Sets new preactivation symbolic bounds (the lower and the upper bound equations).
        """
        assert self.size == new_symbolic_bounds.get_size()
        self.preactivation_symbolic_bounds = new_symbolic_bounds

    def get_preactivation_symbolic_bounds(self):
        return self.preactivation_symbolic_bounds

    def set_symbolic_bounds(self, new_symbolic_bounds):
        """
        Sets new symbolic bounds (the lower and the upper bound equations).
        """
        assert self.size == new_symbolic_bounds.get_size()
        self.symbolic_bounds = new_symbolic_bounds

    def get_symbolic_bounds(self):
        return self.symbolic_bounds

    def set_bounds(self, new_bounds):
        self.bounds = new_bounds

    def get_bounds(self):
        return self.bounds

    def set_preactivation_bounds(self, new_bounds):
        self.preactivation_bounds = new_bounds

    def get_preactivation_bounds(self):
        return self.preactivation_bounds

    def set_encoding(self, node_index, enc):
        assert 0 <= node_index <= self.size - 1

        self.encodings[node_index] = enc

    def get_encodings(self):
        return self.encodings

    def set_status(self, node_index, status):
        assert 0 <= node_index <= self.size - 1

        self.status[node_index] = status

    def get_status(self):
        return self.status

    def get_stable_number(self):
        if self.stable_number != -1:
            return self.stable_number

        if self.status is not None:
            stable_n = 0

            for node_n in range(self.size):
                if self.status[node_n] != ReluNodeStatus.UNSTABLE:
                    stable_n += 1

            self.stable_number = stable_n
            return stable_n
        return -1

    def remove_symbolic_bounds(self):
        """
        Symbolic bounds take a lot of memory since they store matrices
        of size input_layer_size x layer_size.
        For high-dimensional networks these matrices can be very large.

        When we need to store many instances of layers in a list or a queue,
        it is better to get rid of the symbolic bounds.
        """
        self.preactivation_symbolic_bounds = None
        self.symbolic_bounds = None

class NodeVariableNetwork(object):
    """
    A data structure for describing all parameters required for an MILP encoding of a feed-forward neural network.

    Consists of an array of layers, each storing the relevant information
    for the encoding of the nodes of the layers such as:

      * variable names (?)
      * node bounds
      * how the node should be encoded:
        -- full MILP encoding
        -- over-approximation (linear or some other relaxation)
        -- under-approximation (fixing nodes to an active on inactive state)

    """

    INPUT_LAYER_INDEX = 0

    def __init__(self):
        self.layers = []
        self.initialised = False
        self.bounds_computed = False

        self.stable_number = -1
        self.instable_number = -1
        self.stable_numbers_per_layer = []
        self.stable_ratio = -1

        self.bounds_calculator_class_name = None

    def initialise_layers(self, network_model):
        """
        Initialise the layers given the network model.

        Layer 0 corresponds to the input layer.
        Layer 1 corresponds to the layer 0 in the network model.
        Etc.
        """

        if self.initialised:
            raise Exception("The node variable network", self, "has already been initialised")

        self.initialised = True

        layer_count = 0
        self.layers.append(NodeVariableLayer(network_model.input_size, layer_count, False))

        from scripts.encoding.network_model import ReluLayer
        for network_layer in network_model.layers:
            layer_count += 1
            self.layers.append(NodeVariableLayer(network_layer.output_size, layer_count,
                                                 isinstance(network_layer, ReluLayer)))

    def get_stable_number(self):
        """
        Returns statistics about the number of stable nodes in the network
        across all layers
        """
        if self.stable_number != -1:
            return float("{:.4f}".format(self.stable_ratio)), self.stable_number, self.instable_number, self.stable_numbers_per_layer

        stable_n = 0
        total_n = 0
        as_array = []
        for layer in self.layers[1:]:
            l_stable = layer.get_stable_number()
            if l_stable != -1:
                as_array.append(l_stable)
                stable_n += l_stable
                total_n += layer.size

        self.stable_ratio = stable_n/total_n
        self.stable_number = stable_n
        self.instable_number = total_n - stable_n
        self.stable_numbers_per_layer = as_array

        return self.stable_ratio, self.stable_number, self.instable_number, self.stable_numbers_per_layer

    def get_linearly_relaxed_number(self):
        rel_n = 0
        for layer in self.layers[1:]:
            rel_n += layer.get_linearly_relaxed_number()

        return rel_n

    def remove_symbolic_bounds(self):
        """
        Symbolic bounds take a lot of memory since they store matrices
        of size input_layer_size x layer_size.
        For high-dimensional networks these matrices can be very large.

        When we need to store many instances of layers in a list or a queue,
        it is better to get rid of the symbolic bounds.
        """
        for layer in self.layers:
            layer.remove_symbolic_bounds()

    def set_bounds_computed(self):
        self.bounds_computed = True

    def get_bounds_computed(self):
        return self.bounds_computed

    def store_bounds_calculator_class(self, class_name):
        self.bounds_calculator_class_name = class_name

    def get_bounds_calculator_class(self):
        return self.bounds_calculator_class_name
