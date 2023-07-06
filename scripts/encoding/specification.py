import numpy as np
from operator import __le__, __ge__, __eq__

from scripts.encoding.symb_backward_bounds_calc import IntervalBounds

(LT, GT, NE) = ('<', '>', '!=')
(LE, GE, EQ) = ('<=', '>=', '==')

SENSE_MAP = {GE: __ge__, LE: __le__, EQ: __eq__}


class StateCoordinate:
    """
    the class representing a component of an n-ary tuple,
    to be used in VarVarConstraint and VarConstConstraint
    for describing conditions on layers of neural networks
    """

    def __init__(self, i):
        self.i = i

    def __str__(self):
        return "({})".format(self.i)


class Constraint:
    """
    the abstract class for an atomic formula which is a comparison between two terms
    """

    def __init__(self, op1, sense, op2):
        """
        :param op1:
        :param sense: a comparison operator, one of LT, GT, NE, LE, GE, EQ
        :param op2:
        """
        assert sense in {LT, GT, NE, LE, GE, EQ}
        super(Constraint, self).__init__()
        self.op1 = op1
        self.op2 = op2
        self.sense = sense

    def get_atomic_constraint(self, state_vars):
        pass


class VarConstConstraint(Constraint):
    """
    the class representing an inequality between a component and a constant,
    e.g., (0) > 1500 read as the value of the first component is greater than 1500.
    """

    def __init__(self, op1, sense, op2):
        """
        :param op1: an instance of StateCoordinate
        :param sense: a comparison operator (see Constraint)
        :param op2: a number
        """
        assert isinstance(op1, StateCoordinate)
        assert isinstance(op2, int) or isinstance(op2, float)
        super(VarConstConstraint, self).__init__(op1, sense, op2)

    def __str__(self):
        return self.op1.__str__() + self.sense + "{}".format(self.op2)

    def get_atomic_constraint(self, state_vars):
        op1 = state_vars[self.op1.i]
        op2 = self.op2
        if self.sense in SENSE_MAP:
            return SENSE_MAP[self.sense](op1, op2)
        else:
            raise Exception("Unexpected sense", self.sense)


class NAryConjFormula:

    def __init__(self, clauses):
        self.clauses = clauses
        self.name = "AND"

    def __str__(self):
        return self.name + "(" + ",".join([clause.__str__() for clause in self.clauses]) + ")"


class GenericSpecification(object):
    """
    A data structure for encoding a generic specification for a
    neural network verification problem.

    Consists of

    * input bounds - the bounds on the input layer
    * output constraints - the constraints on the ouput layer
    """

    def __init__(self, input_bounds, output_constr):
        """
        :args: input_bounds: interval bounds for the input layer
        :args: output_constr: Boolean formula encoding constraints on the output layer.
        """
        self.input_bounds = input_bounds
        self.output_constr = output_constr

    def get_input_bounds(self):
        return self.input_bounds

    def get_output_constraints(self):
        return self.output_constr

    def clone_new_input_bounds(self, new_input_bounds):
        return GenericSpecification(new_input_bounds, self.output_constr)


class RobustExplanationSpecification(GenericSpecification):
    def __init__(self, input, label, n, radius=1):
        self.n = n

        self.input = input

        self.label = label
        assert label in [0, 1]
        self.desired_label = (int)(1 - label)

        input_lower_bounds = np.maximum(input - radius, 0)
        input_upper_bounds = np.minimum(input + radius, 1)

        output_constr = NAryConjFormula(
            [VarConstConstraint(StateCoordinate(i), EQ, self.desired_label) for i in range(self.n)]
        )

        super(RobustExplanationSpecification, self).__init__(IntervalBounds(input_lower_bounds, input_upper_bounds),
                                                        output_constr)


