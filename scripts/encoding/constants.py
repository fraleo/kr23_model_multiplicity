from enum import Enum


class NeuralOutputSettings(Enum):
    Q_VALUES = 0
    ONE_HOT_ARGMAX = 1
    INTEGER_ARGMAX = 2


class ActivationFunctions(Enum):
    UNKNOWN = 0
    LINEAR = 1
    RELU = 2
    SIGMOID = 3


class RNNUnrollingMethods:
    INPUT_ON_START = 1
    INPUT_ON_DEMAND = 2


NOT_IMPLEMENTED = "Needs to be implemented."


class ReluNodeEncoding(Enum):
    UNDEFINED = 0
    IDENTITY = 1
    ZERO = 2
    BIGM = 3
    LINEAR_RELAX = 4
    QUAD_RELAX = 5
    UNDER_APPROX_POS = 6
    UNDER_APPROX_NEG = 7


class ReluNodeStatus(Enum):
    ACTIVE = 0
    INACTIVE = 1
    UNSTABLE = 2
