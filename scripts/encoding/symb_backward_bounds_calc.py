from enum import Enum

import numpy as np


from scripts.encoding.constants import ReluNodeStatus, ReluNodeEncoding


def compute_lower(weights_minus, weights_plus, input_lower, input_upper):
    return weights_plus.dot(input_lower) + weights_minus.dot(input_upper)


def compute_upper(weights_minus, weights_plus, input_lower, input_upper):
    return weights_plus.dot(input_upper) + weights_minus.dot(input_lower)


class LinearEquation:
    """
    matrix is an (n x m) np array
    offset is an (n) np array

    An object represents n linear functions f(i) of m input variables x

    f(i) = matrix[i]*x + offset[i]

    """
    def __init__(self, matrix, offset):
        self.size = matrix.shape[0]
        self.matrix = matrix
        self.offset = offset

    def clone(self):
        return LinearEquation(self.matrix.copy(), self.offset.copy())

    @staticmethod
    def get_identity(size):
        matrix = np.identity(size, dtype=float)
        offset = np.zeros(size)
        return LinearEquation(matrix, offset)

    @staticmethod
    def get_constant(offset):
        size = len(offset)
        matrix = np.zeros((size, size), dtype=float)
        return LinearEquation(matrix, offset)

    def get_size(self):
        return self.size

    def get_matrix(self):
        return self.matrix

    def get_offset(self):
        return self.offset

    def compute_max_values(self, input_bounds):
        weights_plus = np.maximum(self.matrix, np.zeros(self.matrix.shape))
        weights_minus = np.minimum(self.matrix, np.zeros(self.matrix.shape))
        return compute_upper(weights_minus, weights_plus, input_bounds.get_lower(), input_bounds.get_upper()) + self.offset

    def compute_min_values(self, input_bounds):
        weights_plus = np.maximum(self.matrix, np.zeros(self.matrix.shape))
        weights_minus = np.minimum(self.matrix, np.zeros(self.matrix.shape))
        return compute_lower(weights_minus, weights_plus, input_bounds.get_lower(), input_bounds.get_upper()) + self.offset



class AbstractBounds:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def get_lower(self):
        return self.lower

    def get_upper(self):
        return self.upper


class IntervalBounds(AbstractBounds):

    def __init__(self, lower, upper):
        super(IntervalBounds, self).__init__(lower, upper)

        self.size = len(lower)

    def __repr__(self):
        return ', '.join(["{}:({},{})".format(i,l,u) for (i,l,u) in
                          zip(range(self.size),
                              [float("{:.3f}".format(l)) for l in self.lower],
                              [float("{:.3f}".format(u)) for u in self.upper])])

    def get_size(self):
        return self.size

    def get_upper_bounds(self):
        return self.upper

    def get_lower_bounds(self):
        return self.lower

    def get_dimension_bounds(self, dim):
        assert 0 <= dim < self.size
        return self.lower[dim], self.upper[dim]

    def get_volume(self):
        sides = [u-l for (l,u) in zip(self.lower, self.upper)]
        result = 0
        for x in sides:
            result += x
        return result

class SymbolicLinearBounds(AbstractBounds):

    def __init__(self, lower, upper):
        super(SymbolicLinearBounds, self).__init__(lower, upper)

        self.size = lower.get_size()

    def get_size(self):
        return self.size

    def to_interval_bounds(self, input_bounds):
        return IntervalBounds(self.get_lower_bounds(input_bounds), self.get_upper_bounds(input_bounds))

    def get_upper_bounds(self, input_bounds):
        return self.upper.compute_max_values(input_bounds)

    def get_lower_bounds(self, input_bounds):
        return self.lower.compute_min_values(input_bounds)

    def get_all_bounds(self, input_bounds):
        return self.lower.compute_min_values(input_bounds),\
               self.lower.compute_max_values(input_bounds),\
               self.upper.compute_min_values(input_bounds),\
               self.upper.compute_max_values(input_bounds)

    def to_interval_bounds(self, input_bounds):
        return IntervalBounds(self.lower.compute_min_values(input_bounds),
                              self.upper.compute_max_values(input_bounds))


class LowerBoundRelaxation(Enum):
    """
    Various methods for computing lower bound relaxations for unstable nodes
    (stable nodes always have the same lower bounds, inactive 0, active the identity).
    """

    # The lower bound is either 0, or the preactivation itself.
    ZERO_IDENTITY = 1

    # The lower bound is either 0, or the linear relaxation of the preactivation.
    ZERO_SLOPE = 2

    # The lower bound is the linear relaxation of the preactivation.
    SLOPE = 3

class SymbolicBackwardBoundsCalculator:
    """
    Bounds calculator based on backwards substitution of symbolic bounds
    as defined in the Abstract Domain for Certifying Neural Networks paper.
    (with slight variations on how to choose the symbolic lower bound).

    """

    ROUNDING_PRECISION = 12

    def __init__(self, lower_bound_relaxation=LowerBoundRelaxation.ZERO_SLOPE):
        self.lower_bound_relaxation = lower_bound_relaxation

    def compute_bounds(self, network_model, var_network, input_bounds):

        assert not var_network.get_bounds_computed()

        """
        Initialise the bounds of the input layer (number 0).
        Only concrete, no symbolic bounds.
        """
        var_network.layers[0].set_bounds(input_bounds)

        """
        Compute the bounds of all subsequent layers
        by making use of the bounds computed at the previous step
        """
        for layer_i in range(1, len(var_network.layers)):
            self._compute_layer_bounds(network_model, var_network, layer_i)

        var_network.set_bounds_computed()

    def _compute_layer_bounds(self, network_model, var_network, layer_n):
        """
        Computes bounds of the layer number layer_n in var_network
        from the bounds of the previous layer.
        """
        assert 1 <= layer_n <= len(var_network.layers) - 1

        """
        The network layer containing the current weights and bias
        """
        network_layer = network_model.get_layer(layer_n)

        from scripts.encoding.network_model import TransformationLayer
        if isinstance(network_layer, TransformationLayer):
            self._compute_transformation_layer_bounds(network_model, var_network, layer_n)

        else:
            raise Exception("Unexpected NetworkLayer" + network_layer)

    def _compute_transformation_layer_bounds(self, network_model, var_network, layer_n, fixed=None):
        """
        Computes bounds of the layer number layer_n in var_network
        from the bounds of the previous layer.
        The layer is either DenseReluLayer or LinearLayer.
        """
        assert 1 <= layer_n <= len(var_network.layers) - 1

        """
        The network layer containing the current weights and bias
        """
        network_layer = network_model.get_layer(layer_n)

        """
        We will set the bounds of the current variable layer.
        """
        current_layer = var_network.layers[layer_n]

        """
        First get the lower and upper bound equations 
        corresponding to the linear transformation of the layer
        """
        linear_transformation_eq = network_layer.as_linear_equation()

        preact_lower_eq = linear_transformation_eq
        preact_upper_eq = linear_transformation_eq

        """
        Set the preactivation bounds of the current layer
        """
        preact_lower_eq_input, preact_lower_bounds = self._get_concrete_bound(preact_lower_eq, var_network, layer_n, "lower")
        preact_upper_eq_input, preact_upper_bounds = self._get_concrete_bound(preact_upper_eq, var_network, layer_n, "upper")

        current_layer.set_preactivation_bounds(IntervalBounds(preact_lower_bounds, preact_upper_bounds))

        """
        Set the preactivation symbolic bounds that are equations of the input layer variables
        """
        current_layer.set_preactivation_symbolic_bounds(SymbolicLinearBounds(preact_lower_eq_input, preact_upper_eq_input))

        """
        Then compute the lower and upper bound equations 
        corresponding to the relaxation of the activation function
        """
        (postact_lower_eq, postact_lower_bounds), (postact_upper_eq, postact_upper_bounds) = \
            self._apply_activation(preact_lower_eq, preact_upper_eq, preact_lower_bounds, preact_upper_bounds, network_layer)

        """ 
        Set the computed symbolic bounds to the layer number layer_n
        (takes into account the linear transformation and the activation function
        to optimise bounds computation).
        
        These equations are equations of the variables from the previous layer,
        not input variables.
        """
        current_layer.set_symbolic_bounds(SymbolicLinearBounds(postact_lower_eq, postact_upper_eq))

        """
        Set the concrete bounds.
        It is important that the lower bounds for postactivations are at least 0!
        """
        from scripts.encoding.network_model import ReluLayer
        if isinstance(network_layer, ReluLayer):
            postact_lower_bounds = np.maximum(postact_lower_bounds, np.zeros(current_layer.size))

            # Set status and encoding of the relu nodes
            for i in range(current_layer.size):
                if preact_lower_bounds[i] >= 0:
                    current_layer.set_status(i, ReluNodeStatus.ACTIVE)
                    current_layer.set_encoding(i, ReluNodeEncoding.IDENTITY)
                elif preact_upper_bounds[i] <= 0:
                    current_layer.set_status(i, ReluNodeStatus.INACTIVE)
                    current_layer.set_encoding(i, ReluNodeEncoding.ZERO)
                else:
                    current_layer.set_status(i, ReluNodeStatus.UNSTABLE)
                    current_layer.set_encoding(i, ReluNodeEncoding.BIGM)

        current_layer.set_bounds(IntervalBounds(postact_lower_bounds, postact_upper_bounds))

    def _compute_flatten_layer_bounds(self, network_model, var_network, layer_n):
        """
        Computes bounds of the layer number layer_n in var_network
        from the bounds of the previous layer.

        The bounds are already stored in a "flat" way.
        So nothing to be done.
        """
        assert 1 <= layer_n <= len(var_network.layers) - 1

        prev_layer = var_network.layers[layer_n - 1]
        current_layer = var_network.layers[layer_n]

        concrete_bounds = prev_layer.get_bounds()

        current_layer.set_preactivation_bounds(concrete_bounds)
        current_layer.set_bounds(concrete_bounds)
        current_layer.set_symbolic_bounds(
            SymbolicLinearBounds(LinearEquation.get_identity(current_layer.size),
                                 LinearEquation.get_identity(current_layer.size)))
    def _apply_activation(self, preact_lower_eq, preact_upper_eq, preact_lower_bounds, preact_upper_bounds, network_layer):
        from scripts.encoding.network_model import ReluLayer, LinearLayer
        if isinstance(network_layer, ReluLayer):
            return self.get_relu_relax_lower_bound_equation(preact_lower_eq, preact_lower_bounds, preact_upper_bounds), \
                   self.get_relu_relax_upper_bound_equation(preact_upper_eq, preact_lower_bounds, preact_upper_bounds)
        elif isinstance(network_layer, LinearLayer):
            return (preact_lower_eq, preact_lower_bounds), \
                   (preact_upper_eq, preact_upper_bounds)

        raise Exception("Not supporting linear relaxation for layer", network_layer)

    def get_relu_relax_lower_bound_equation(self, preact_lower_eq, preact_lower_bounds, preact_upper_bounds):
        """
        Compute the resulting lower bound equation after relaxing ReLU,
        qiven a preactivation lower bound equation.
        """
        if self.lower_bound_relaxation == LowerBoundRelaxation.ZERO_IDENTITY:
            postact_lower_eq, postact_lower_bounds = \
                self._lower_bound_relax_zero_identity(preact_lower_eq, preact_lower_bounds, preact_upper_bounds)

        elif self.lower_bound_relaxation == LowerBoundRelaxation.ZERO_SLOPE:
            postact_lower_eq, postact_lower_bounds = \
                self._lower_bound_relax_zero_slope(preact_lower_eq, preact_lower_bounds, preact_upper_bounds)

        else:
            raise Exception("Unexpected value of lower bound relaxation method", self.lower_bound_relaxation)

        return postact_lower_eq, postact_lower_bounds

    def get_relu_relax_upper_bound_equation(self, preact_upper_eq, preact_lower_bounds, preact_upper_bounds):
        return self._upper_bound_relax_slope(preact_upper_eq, preact_lower_bounds, preact_upper_bounds)

    def _get_concrete_bound(self, equation, var_network, layer_n, end):
        """
        Given an equation for the current layer (which depends on the variables of the previous layer),
        computes the lower of the upper bound equation from the variables of the input layer
        by backwards substitution of the equations of the previous layers.

        Then, computes the concrete bounds of the obtained equation.

        end indicates if we want to compute the lower or the upper bound.
        """
        current_layer = layer_n - 1

        current_matrix = equation.get_matrix()
        current_offset = equation.get_offset()

        while current_layer > 0:
            prev_equation = var_network.layers[current_layer].get_symbolic_bounds()
            current_matrix, current_offset = self._substitute_one_step_back(current_matrix, current_offset, prev_equation, end)

            current_layer -= 1

        equation_input = LinearEquation(current_matrix, current_offset)

        if end == "lower":
            # self._round_down(current_matrix)
            # self._round_down(current_offset)
            bound = equation_input.compute_min_values(var_network.layers[0].get_bounds())
        else:
            # self._round_up(current_matrix)
            # self._round_up(current_offset)
            bound = equation_input.compute_max_values(var_network.layers[0].get_bounds())

        return equation_input, bound

    def _substitute_one_step_back(self, current_matrix, current_offset, prev_equations, end):
        """
        Performs one substitution step.

        Given an equation mapping R^n -> R^m in the form of a matrix and an offset, and
        previous equations mapping R^k to R^n,
        computes a new equation (in the form of a matrix and an offset) that
        maps R^k to R^m.
        """
        prev_lower_eq = prev_equations.get_lower()
        prev_upper_eq = prev_equations.get_upper()

        matrix_pos = np.maximum(current_matrix, np.zeros(current_matrix.shape))
        matrix_neg = np.minimum(current_matrix, np.zeros(current_matrix.shape))

        if end == "lower":
            current_matrix = matrix_pos.dot(prev_lower_eq.get_matrix()) + matrix_neg.dot(prev_upper_eq.get_matrix())
            current_offset = matrix_pos.dot(prev_lower_eq.get_offset()) + matrix_neg.dot(prev_upper_eq.get_offset()) + \
                             current_offset

            # self._round_down(current_matrix)
            # self._round_down(current_bias)
        else:
            current_matrix = matrix_pos.dot(prev_upper_eq.get_matrix()) + matrix_neg.dot(prev_lower_eq.get_matrix())
            current_offset = matrix_pos.dot(prev_upper_eq.get_offset()) + matrix_neg.dot(prev_lower_eq.get_offset()) + \
                             current_offset

            # self._round_up(current_matrix)
            # self._round_up(current_bias)

        return current_matrix, current_offset

    def _lower_bound_relax_zero_identity(self, preact_lower_eq, preact_lower_bounds, preact_upper_bounds):
        """
        The lower bound of unstable nodes is either 0, or
        the preactivation itself (hence, the identity).

        The latter is the case when the upper bound is greater than or equal to
        the absolute value of the lower bound,
        thus resulting in a triangle of smaller area than the one formed by 0.

        The former is the case when the absolute value of the lower bound is greater than the upper bound,
        thus resulting in a triangle of smaller area than the one formed by the identity.

        See the Abstract Domain for Certifying Neural Networks paper for more explanations.
        """
        size = len(preact_lower_bounds)

        # matrix and offset for the relaxation
        matrix = np.zeros(preact_lower_eq.get_matrix().shape)
        offset = np.zeros(preact_lower_eq.get_offset().shape)

        postact_lower_bounds = np.array(preact_lower_bounds)

        for i in range(size):
            if preact_lower_bounds[i] >= 0 or preact_upper_bounds[i] >= -preact_lower_bounds[i]:
                # the lower bound is exactly the preactivation
                matrix[i] = preact_lower_eq.get_matrix()[i]
                offset[i] = preact_lower_eq.get_offset()[i]
            else: # upper[i] <= 0 (inactive node)
                  # or
                  # -lower[i] > upper[i]
                # lower bound is 0
                postact_lower_bounds[i] = 0

        return LinearEquation(matrix, offset), postact_lower_bounds

    def _lower_bound_relax_zero_slope(self, preact_lower_eq, preact_lower_bounds, preact_upper_bounds):
        """
        The lower bound of unstable nodes is either 0, or
        the linear relaxation of the preactivation (hence, the slope).

        The latter is the case when the upper bound is greater than or equal to the absolute value of the lower bound,
        thus resulting in a triangle of smaller area than the one formed by 0.

        The former is the case when the absolute value of the lower bound is greater than the upper bound,
        thus resulting is a triangle of smaller area than the one formed by the slope.
        """
        size = len(preact_lower_bounds)

        # matrix and offset for the relaxation
        matrix = np.zeros(preact_lower_eq.get_matrix().shape)
        offset = np.zeros(preact_lower_eq.get_offset().shape)

        postact_lower_bounds = np.array(preact_lower_bounds)

        for i in range(size):
            if preact_lower_bounds[i] >= 0:
                # the lower bound is exactly the preactivation
                matrix[i] = preact_lower_eq.get_matrix()[i]
                offset[i] = preact_lower_eq.get_offset()[i]
            elif preact_upper_bounds[i] >= -preact_lower_bounds[i]:
                # Unstable node, lower bound is linear relaxation of the equation
                k = preact_upper_bounds[i] / (preact_upper_bounds[i] - preact_lower_bounds[i])
                matrix[i] = preact_lower_eq.get_matrix()[i] * k
                offset[i] = preact_lower_eq.get_offset()[i] * k
                postact_lower_bounds[i] *= k
            else: # upper[i] <= 0 (inactive node)
                  # or
                  # -lower[i] > upper[i]
                # lower bound is 0
                  postact_lower_bounds[i] = 0

        return LinearEquation(matrix, offset), postact_lower_bounds

    def _lower_bound_relax_slope(self, preact_lower_eq, preact_lower_bounds, preact_upper_bounds):
        """
        The lower bound of unstable nodes is the linear relaxation of the preactivation (hence, the slope)
        as is defined in the Neurify paper.
        """
        size = len(preact_lower_bounds)

        # matrix and offset for the relaxation
        matrix = np.zeros(preact_lower_eq.get_matrix().shape)
        offset = np.zeros(preact_lower_eq.get_offset().shape)

        postact_lower_bounds = np.array(preact_lower_bounds)

        for i in range(size):
            if preact_lower_bounds[i] >= 0:
                # the lower bound is exactly the preactivation
                matrix[i] = preact_lower_eq.get_matrix()[i]
                offset[i] = preact_lower_eq.get_offset()[i]
            elif preact_lower_bounds[i] < 0 and preact_upper_bounds[i] > 0:
                # Unstable node, lower bound is linear relaxation of the equation
                k = preact_upper_bounds[i] / (preact_upper_bounds[i] - preact_lower_bounds[i])
                matrix[i] = preact_lower_eq.get_matrix()[i] * k
                offset[i] = preact_lower_eq.get_offset()[i] * k
                postact_lower_bounds[i] *= k
            else: # upper[i] <= 0 (Inactive node)
                # lower bound is 0
                postact_lower_bounds[i] = 0

        return LinearEquation(matrix, offset), postact_lower_bounds

    @staticmethod
    def _upper_bound_relax_slope(preact_upper_eq, preact_lower_bounds, preact_upper_bounds):
        """
        Compute the resulting upper bound equation after relaxing ReLU,
        qiven a preactivation upper bound equation.

        input_bounds are required for computing the concrete bounds.
        """
        size = len(preact_lower_bounds)

        # matrix and offset for the relaxation
        matrix = np.zeros(preact_upper_eq.get_matrix().shape)
        offset = np.zeros(preact_upper_eq.get_offset().shape)

        postact_upper_bounds = np.array(preact_upper_bounds)
        for i in range(size):
            if preact_lower_bounds[i] >= 0:
                # the upper bound is exactly the preactivation
                matrix[i] = preact_upper_eq.get_matrix()[i]
                offset[i] = preact_upper_eq.get_offset()[i]
            elif preact_upper_bounds[i] >= 0:
                # Unstable node - linear relaxation of preactivation
                k = preact_upper_bounds[i] / (preact_upper_bounds[i] - preact_lower_bounds[i])
                matrix[i] = k * preact_upper_eq.get_matrix()[i]
                offset[i] = (preact_upper_eq.get_offset()[i] - preact_lower_bounds[i]) * k
            else: # preact_upper_bounds[i] <= 0 (inactive node)
                # The upper bound is 0
                postact_upper_bounds[i] = 0

        return LinearEquation(matrix, offset), postact_upper_bounds

